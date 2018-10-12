from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import Imputer, StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# read data; here, from local files
df_numeric = spark.read.csv('train_numeric.csv',header=True,inferSchema="true")
df_categorical = spark.read.csv('train_categorical.csv',header=True,inferSchema="true")
df_date = spark.read.csv('train_date.csv',header=True,inferSchema="true")

# take the names of the variables that are used as input
num_input_vars = list(set(df_numeric.columns) - {'Id','Response'})
cat_input_vars = list(set(df_categorical.columns) - {'Id'})
date_input_vars = list(set(df_date.columns) - {'Id'})

# The pyspark imputer does not work nicely with catergorical values,
# so we fill missing values with a constant string.
# NB: Do not use empty strings for imputations, it will likely cause problems later.
df_categorical = df_categorical.fillna('unknown')

# combine the data frames, note that we drop duplicate variables
df_all = df_numeric.join(df_categorical,df_numeric.Id==df_categorical.Id).drop(df_categorical.Id)
df_all = df_all.join(df_date,df_all.Id==df_date.Id).drop(df_date.Id)

# Imputate numerical variables. We use median for imputation, though it is likely not
# really the smartest choice here, especially with the date variables.
imputer = Imputer(
    inputCols=num_input_vars+date_input_vars,
    outputCols=num_input_vars+date_input_vars,
    strategy='median')
df_all = imputer.fit(df_all).transform(df_all)

# In order to use categorical variables, they need to be first indexed
# and then encoded (i.e. dummified). This needs to be done for each variable seperately.
# Because of this, we create indexers and encoders for each categorical variable.
cat_indexers = [
    StringIndexer(inputCol=c, outputCol="{0}_i".format(c),handleInvalid='keep')
    for c in cat_input_vars
]

cat_encoders = [
    OneHotEncoder(
        inputCol=indexer.getOutputCol(),
        outputCol="{0}_e".format(indexer.getOutputCol()))
    for indexer in cat_indexers
]

# these are the encoded categorical variable names
cat_enc_vars = [encoder.getOutputCol() for encoder in cat_encoders]

# here's a complete list of input variables for the model
input_vars = num_input_vars+date_input_vars+cat_enc_vars

# VectorAssembler takes a set of variables and combines them
# into a single input variable
assembler = VectorAssembler(inputCols=input_vars,outputCol="features")

# and here's our prediction model; let us just use a random forest here
rfc = RandomForestClassifier(labelCol="Response",featuresCol="features",seed=777)

# Now we can build a pipeline from the feature transforms adn the random forest:
pipeline = Pipeline(stages=cat_indexers + cat_encoders+[assembler,rfc])

# With random forests, you can naturally use Out-of-Bag error values
# for model tuning. Here, just to show how to use the cross validator
# with a pipeline for hyperparamter tuning.

# straightforward classifier evaluator
evaluator = MulticlassClassificationEvaluator(
    labelCol="Response", predictionCol="prediction", metricName="accuracy")

# as an example of parameter hypertuning, we will try two forest sizes
paramGrid = ParamGridBuilder().addGrid(rfc.numTrees, [10, 20]).build()

# the cross validator; run ten folds here
crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=10)

# Finally, train the model via pipeline and cross validatorself.
# The fit functions returns the best model according to the cross validator.
# NB: with full data, this will take a lenghty time!
cv_model = crossval.fit(df_all)

# So, now that the model is trained, we can to some predictions to the test data.
# First, similar process as with the training data.
df_numeric_test = spark.read.csv('test_numeric.csv',header=True,inferSchema="true")
df_categorical_test = spark.read.csv('test_categorical.csv',header=True,inferSchema="true")
df_date_test = spark.read.csv('test_date.csv',header=True,inferSchema="true")
df_categorical_test = df_categorical_test.fillna('unknown')
df_all_test = df_numeric_test.join(
    df_categorical_test,df_numeric_test.Id==df_categorical_test.Id).drop(df_categorical_test.Id)
df_all_test = df_all_test.join(
    df_date_test,df_all_test.Id==df_date_test.Id).drop(df_date_test.Id)
df_all_test = imputer.fit(df_all_test).transform(df_all_test)

# then. the predictions
predictions = cv_model.transform(df_all_test)

# select the output from the prediction df, rename "prediction" as "Response" to match Kaggle format
result = predictions.selectExpr("Id as Id", "prediction as Response")

# write the result file.
# NB: writes a directory called result where the cryptically named .csv file(s) will appear
result.select('Id',result.Response.astype('int')).write.option('header','true').csv('../ex7_data/result')
