import logging
import argparse
import sys

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, RegexTokenizer, StopWordsRemover, CountVectorizer, HashingTF, IDF
from pyspark.ml.clustering import LDA
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
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)

    ########################
    # READING RAW FILES

    logger.info('Creating base df.')
    df = spark.read.json(args.input)

    df = df.dropDuplicates(['doi'])
    if args.sample:
        df = df.sample(False, float(args.sample), 42)
    df.cache()
    if args.debug:
        df.printSchema()
        df.explain(True)
    logger.info('Base df created, papers in sample: %d' % df.count())

    #########################
    # SPLITTING DF INTO METADATA AND FULLTEXT

    fulltexts = df.select('doi', 'abstract', 'fulltext')

    #########################
    # DEFINING TRANSFORMERS and ESTIMATORS

    logger.info('Initializing feature pipeline.')

    stringconcat = StringConcatenator(inputCols=["abstract", "fulltext"],
                                      outputCol="texts")
    noun_extractor = NounExtractor(inputCol='texts', outputCol='filtered')
    selector = ColumnSelector(outputCols=["doi", "filtered"])
    countV = CountVectorizer(inputCol="filtered", outputCol="rawFeatures")
    hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures",
                          numFeatures=5000)
    idf = IDF(inputCol="rawFeatures", outputCol="features")

    idf_pipeline = Pipeline(stages=[stringconcat,
                                    noun_extractor,
                                    selector,
                                    hashingTF,
                                    idf])
    logger.info('Fitting feature pipeline.')
    idf_pipeline_model = idf_pipeline.fit(fulltexts)
    logger.info('Applying feature pipeline.')
    df = idf_pipeline_model.transform(fulltexts)
    df.cache()

    #########################
    # SETTING UP BROADCAST VARIABLES

    # NO BC VARIABLES THIS TIME
    #########################

    if args.eval:
        logger.info('Initializing LDA-model evaluation.')
        # paramGrid = ParamGridBuilder().addGrid(lda.k, [10,50,100])
        train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
        evals = []
        for k in [150, 300, 500, 1000]:
            logger.info('Now training with k=%d.' % k)
            lda = LDA(k=k, maxIter=int(args.maxIter))
            lda_model = lda.fit(train_df)
            logger.info('Now getting metrics for k=%d.' % k)
            evals.append((k,
                          lda_model.logLikelihood(test_df),
                          lda_model.logPerplexity(test_df)))
        pdf = pd.DataFrame.from_records(evals, columns=['k',
                                                        'logLikelihood',
                                                        'logPerplexity'])
        pdf.to_csv(args.output)
        logger.info('Ending workflow, shutting down.')
        sc.stop()
        sys.exit()

    # RUN LDA model
    logger.info('Fitting LDA-model.')
    lda = LDA(k=int(args.k), maxIter=int(args.maxIter))
    lda_model = lda.fit(df)
    logger.info('Vocabulary size of LDA model: %d' % lda_model.vocabSize())
    df = lda_model.transform(df)
    df.write.save(args.output+"df.parquet")

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
    args = parser.parse_args()
    main(args)
