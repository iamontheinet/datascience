// Databricks notebook source
// Import libraries
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.regression.RandomForestRegressor // NOT RandomForestRegressorModel
import org.apache.spark.ml.evaluation.{RegressionEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.classification.RandomForestClassifier // NOT RandomForestClassifierModel
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import com.databricks.ml.local.{ModelExport, ModelFactory}

// COMMAND ----------

// Load advertising data
// The dataset contains advertising budgets (in 1000s of USD) for different media channels (TV, radio and newspaper) and their sales (in 1000s of units).
var df = sqlContext
  .read.format("csv")
  .option("header", "true")
  .option("inferSchema", "true")
  .load("/FileStore/tables/Advertising.csv").drop("_c0")

// Note: for large datasets inferSchema should be set to false so Spark doesn't read every single value in every row to infer its datatype

// COMMAND ----------

// Examine advertising data
df.show()

// COMMAND ----------

df.describe().show()

// COMMAND ----------

// MUST for Spark target
df = df.withColumnRenamed("Sales", "label")

// COMMAND ----------

// MUST for Spark features
var assembler = new VectorAssembler()
  .setInputCols(Array("TV", "Radio", "Newspaper"))
  .setOutputCol("features")

// COMMAND ----------

// Transform features
df = assembler.transform(df)

// COMMAND ----------

df.show()

// COMMAND ----------

// Split dataset into "train" (80%) and "test" (20%) sets -- this is equivalent to train test split in scikit-learn
var Array(train, test) = df.randomSplit(Array(.8, .2), 42)

// COMMAND ----------

train.show(5)

// COMMAND ----------

test.show(5)

// COMMAND ----------

dbutils.widgets.text("NUM_OF_TREES", "30")
var numOfTrees = dbutils.widgets.get("NUM_OF_TREES").toInt
//println(s"numOfTrees : ${numOfTrees}")

// Create RandomForestRegressor instance with hyperparameters "number of trees" set to the default value (30) or parameter passed in and "tree depth" set to 2. 
// NOTE: Hyperparameter values are hard-coded arbitraryly for the sake of illustration. In production, hyperparameters should be tuned to create more accurate models.)
var rf = new RandomForestRegressor().setNumTrees(numOfTrees).setMaxDepth(2)

// COMMAND ----------

// Setup pipeline
var pipeline = new Pipeline().setStages(Array(rf))

// COMMAND ----------

// Setup hyperparams grid
var paramGrid = new ParamGridBuilder().build()

// COMMAND ----------

// Setup model evaluators
var rmseevaluator  = new RegressionEvaluator() // Note: By default, it will show how many units off in the same scale as the target -- RMSE
var r2evaluator  = new RegressionEvaluator().setMetricName("r2") // // Selet R2 as our main scoring metric

// COMMAND ----------

// Setup cross validator
var cv = new CrossValidator()
  .setEstimator(pipeline)
  .setEstimatorParamMaps(paramGrid)
  .setEvaluator(r2evaluator)

// COMMAND ----------

// Fit model on trianing data
var cvModel = cv.fit(train)

// Get the best model based on CrossValidator
val model = cvModel.bestModel.asInstanceOf[PipelineModel]

// Run inference on test dataset
var predictions = model.transform(test)

println(s"RMSE     : ${rmseevaluator.evaluate(predictions)}")
println(s"R2 Score : ${r2evaluator.evaluate(predictions)}")

// COMMAND ----------

// Examine predictions
predictions.show(5)

// COMMAND ----------

// Export and save model to AWS S3
// var awsAccessKeyId = "YOUR_awsAccessKeyId"
// var awsSecretAccessKey = "YOUR_awsSecretAccessKey"
// var awsBucket = "YOUR_awsBucket"

// sc.hadoopConfiguration.set("fs.s3n.awsAccessKeyId", awsAccessKeyId)
// sc.hadoopConfiguration.set("fs.s3n.awsSecretAccessKey", awsSecretAccessKey)

// ModelExport.exportModel(model, awsBucket)

// COMMAND ----------


