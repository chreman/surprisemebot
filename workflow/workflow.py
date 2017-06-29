import logging
import argparse
import sys

from pyspark import SparkContext, SparkConf

from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, StringType, IntegerType, MapType
from pyspark.sql.functions import udf

from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, RegexTokenizer, StopWordsRemover, CountVectorizer, HashingTF, IDF
from pyspark.ml.clustering import LDA
from pyspark.mllib.linalg.distributed import RowMatrix, IndexedRowMatrix, BlockMatrix
from pyspark.mllib.linalg import SparseVector, Vectors
from pyspark.mllib.linalg.distributed import IndexedRow

from transformers import StringListAssembler, StringConcatenator, ColumnSelector, SentTokenizer, ColumnExploder, NounExtractor


def main(args):
    logging.basicConfig(level=logging.INFO, datefmt="%Y/%m/%d %H:%M:%S")
    formatter = logging.Formatter('%(asctime)-15s %(name)s [%(levelname)s] '
                                  '%(message)s')
    logger = logging.getLogger('WorkflowLogger')
    if args.logfile:
        fh = logging.FileHandler(args.logfile)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    logger.info('Beginning workflow')

    conf = SparkConf()
    if (args.awsAccessKeyID and args.awsSecretAccessKey):
        conf.set("spark.hadoop.fs.s3.awsAccessKeyID",
                 args.awsAccessKeyID)
        conf.set("spark.hadoop.fs.s3.awsSecretAccessKey",
                 args.awsSecretAccessKey)
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)

    ########################
    # READING RAW FILES

    logger.info('Creating base df.')
    # df = spark.read.json(args.input)
    #
    # df = df.dropDuplicates(['doi'])
    # if args.sample:
    #     df = df.sample(False, float(args.sample), 42)
    # df.cache()
    # if args.debug:
    #     df.printSchema()
    #     df.explain(True)
    # logger.info('Base df created, papers in sample: %d' % df.count())
    #
    # #########################
    # # SPLITTING DF INTO METADATA AND FULLTEXT
    #
    # fulltexts = df.select('doi', 'abstract', 'fulltext')
    #
    # #########################
    # # DEFINING TRANSFORMERS and ESTIMATORS
    #
    # logger.info('Initializing feature pipeline.')
    #
    # stringconcat = StringConcatenator(inputCols=["abstract", "fulltext"],
    #                                   outputCol="texts")
    # noun_extractor = NounExtractor(inputCol='texts', outputCol='filtered')
    # selector = ColumnSelector(outputCols=["doi", "filtered"])
    # countV = CountVectorizer(inputCol="filtered", outputCol="rawFeatures")
    # hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures",
    #                       numFeatures=5000)
    # idf = IDF(inputCol="rawFeatures", outputCol="features")
    #
    # idf_pipeline = Pipeline(stages=[stringconcat,
    #                                 noun_extractor,
    #                                 selector,
    #                                 hashingTF,
    #                                 idf])
    # logger.info('Fitting feature pipeline.')
    # idf_pipeline_model = idf_pipeline.fit(fulltexts)
    # logger.info('Applying feature pipeline.')
    # df = idf_pipeline_model.transform(fulltexts)
    # df.cache()
    #
    # #########################
    # # SETTING UP BROADCAST VARIABLES
    # index2doi = {i: doi[0]
    #              for i, doi in enumerate(df.select('doi')
    #                                        .rdd.collect())}
    # doi2index = {v: k for k, v in index2doi.items()}
    # I2D = sc.broadcast(index2doi)
    # D2I = sc.broadcast(doi2index)

    # if args.eval:
    #     logger.info('Initializing LDA-model evaluation.')
    #     # paramGrid = ParamGridBuilder().addGrid(lda.k, [10,50,100])
    #     train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    #     evals = []
    #     for k in [150, 300, 500, 1000]:
    #         logger.info('Now training with k=%d.' % k)
    #         lda = LDA(k=k, maxIter=int(args.maxIter))
    #         lda_model = lda.fit(train_df)
    #         logger.info('Now getting metrics for k=%d.' % k)
    #         evals.append((k,
    #                       lda_model.logLikelihood(test_df),
    #                       lda_model.logPerplexity(test_df)))
    #     pdf = pd.DataFrame.from_records(evals, columns=['k',
    #                                                     'logLikelihood',
    #                                                     'logPerplexity'])
    #     pdf.to_csv(args.output)
    #     logger.info('Ending workflow, shutting down.')
    #     sc.stop()
    #     sys.exit()

    # logger.info('Fitting LDA-model.')
    # lda = LDA(k=int(args.k), maxIter=int(args.maxIter))
    # lda_model = lda.fit(df)
    # logger.info('Vocabulary size of LDA model: %d' % lda_model.vocabSize())
    # df = lda_model.transform(df)
    # df.write.json(args.output+"ldadf")
    # sc.stop()
    # sys.exit()

    ###########################
    # SIMILARITY COMPUTATION

    df = spark.read.json(args.input)
    logger.info('Creating topicDistRDD.')
    topicDistRdd = (df.select('topicDistribution')
                      .rdd.coalesce(8)
                      .map(lambda x: x[0].values))
    indexedTopicDistRdd = (topicDistRdd.zipWithIndex()
                                       .map(lambda x: (x[1], x[0])))
    mat = IndexedRowMatrix(indexedTopicDistRdd)

    sims = mat.toCoordinateMatrix()
    sims = (sims.transpose()
                .toIndexedRowMatrix()
                .toRowMatrix()
                .columnSimilarities(threshold=0.8)
                .toRowMatrix())
    sims_num_rows = sims.numRows()
    sims_num_cols = sims.numCols()

    if args.debug:
        logger.info('num_rows: %d' % sims_num_rows)
        logger.info('num_cols: %d' % sims_num_cols)

    n = Vectors.sparse(sims_num_cols, [], [])
    new_rows = sc.parallelize([n]*(sims_num_cols-sims_num_rows))
    sims = sims.rows.union(new_rows)
    logger.info('Creating sims_df.')
    sims = IndexedRowMatrix(sims.zipWithIndex().map(lambda x: (x[1], x[0])))

    sims_num_rows = sims.numRows()
    sims_num_cols = sims.numCols()

    if args.debug:
        logger.info('num_rows: %d' % sims_num_rows)
        logger.info('num_cols: %d' % sims_num_cols)

    logger.info('Creating sims_df.')
    sims_df = sims.rows.toDF()
    if args.debug:
        sims_df.show()
        logger.info('%d' % sims_df.count())

    # Transposing matrix to enable creation of symmetric matrix
    # columnSimilarities returns a upper-triangle matrix
    logger.info('Transposing sims.')
    sims_T = sims.toBlockMatrix().transpose().toIndexedRowMatrix()
    n = IndexedRow(0, Vectors.sparse(sims_num_cols, [], []))
    nrow = sc.parallelize([n]*(sims_num_cols-sims_num_rows))
    sims_T_rdd = nrow.union(sims_T.rows)
    sims_T_IRM = IndexedRowMatrix(sims_T_rdd)
    logger.info('Creating transposed similarities df.')
    sims_T_df = sims_T_IRM.rows.toDF().withColumnRenamed('vector', 'vector2')
    if args.debug:
        sims_T_df.show()
        logger.info('%d' % sims_T_df.count())

    #########################
    # ADDING SPARSE VECTORS TO GET SYMMETRIC MATRIX

    logger.info('Creating complete similarities df.')
    df = sims_df.join(sims_T_df, on="index").sort('index', ascending=True)

    def sum_vectors(row):
        from collections import defaultdict
        index, v1, v2 = row
        if not isinstance(v2, SparseVector):
            new_inds = [i for i, e in enumerate(v2) if e != 0]
            new_vals = [v for v in v2 if v != 0]
            v2 = Vectors.sparse(len(v2), new_inds, new_vals)
        # assert isinstance(v1, SparseVector) and isinstance(v2, SparseVector)
        assert v1.size == v2.size
        values = defaultdict(float)  # Dictionary with default value 0.0
        # Add values from v1
        for i in range(v1.indices.size):
            values[v1.indices[i]] += v1.values[i]
        # Add values from v2
        for i in range(v2.indices.size):
            values[v2.indices[i]] += v2.values[i]
        return Vectors.sparse(v1.size, dict(values))

    df.show()
    results_rdd = df.rdd.map(sum_vectors)
    logger.info('Creating results df.')
    df = IndexedRowMatrix(results_rdd.zipWithIndex()
                                     .map(lambda x: (x[1], x[0]))).rows.toDF()
    df.cache()

    ##########################
    # FIND TOP SIMILARITIES

    def get_top_similarities(sv):
        def getKey(item):
            return item[1]
        ind = sv.indices
        val = sv.values
        similarities = [(i, v)
                        for i, v in zip(ind, val)
                        if v > 0.9]
        sorted_similarities = sorted(similarities, key=getKey)
        return [int(s[0])
                for s in sorted_similarities[:10]]
    udf_top_similarities = udf(get_top_similarities, ArrayType(IntegerType()))

    logger.info('Finding top similarities.')
    df = df.withColumn('top_n', udf_top_similarities(df['vector']))

    def map_index2doi(top_n):
        if type(top_n) != list:
            top_n = [top_n]
        mapper = I2D.value
        return ["http://dx.doi.org/"+mapper.get(i) for i in top_n]
    udf_mapindex2doi = udf(map_index2doi, ArrayType(StringType()))

    df = df.withColumn('rec_dois', udf_mapindex2doi(df['top_n']))
    df = df.withColumn('doi', udf_mapindex2doi(df['index']))

    def make_recs(doi, rec_dois):
        return {doi[0]: rec_dois}
    udf_make_recs = udf(make_recs,
                        MapType(StringType(),
                                ArrayType(StringType())))

    df = df.select(udf_make_recs(df['doi'], df['rec_dois']).alias('recs'))

    logger.info('Exporting results.')
    df.write.json(args.output)

    # with open(path.join(args.output, "mappings.json"), 'w') as outfile:
    #     for r in df.rdd.collect():
    #         outfile.write(json.dumps(r.asDict().get('recs'))+'\n')

    logger.info('Ending workflow, shutting down.')
    sc.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='do stuff')
    parser.add_argument('--input', dest='input', help='relative or absolute '
                        'path of the input folder')
    parser.add_argument('--output', dest='output', help='relative or absolute '
                        'path of the output folder')
    parser.add_argument('--k', dest='k', help='number of topics')
    parser.add_argument('--maxIter', dest='maxIter', help='maximum number of '
                        'iterations')
    parser.add_argument('--eval', dest='eval', help='flag for evaluation mode',
                        action='store_true')
    parser.add_argument('--sample', dest='sample', help='fraction of dataset '
                        'to sample, e.g. 0.01')
    parser.add_argument('--debug', dest='debug', help='flag for debug mode, '
                        'rdds now evaluated greedy', action='store_true')
    parser.add_argument('--logfile', dest='logfile', help='relative or '
                        'absolute path of the logfile')
    parser.add_argument('--awsAccessKeyID', dest='awsAccessKeyID',
                        help='awsAccessKeyID')
    parser.add_argument('--awsSecretAccessKey', dest='awsSecretAccessKey',
                        help='awsSecretAccessKey')
    args = parser.parse_args()
    main(args)
