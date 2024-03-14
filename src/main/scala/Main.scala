import breeze.linalg.Matrix
import org.apache.spark.rdd.RDD

import scala.io._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd._
import org.apache.log4j.Logger
import org.apache.log4j.Level

import scala.collection._
import org.apache.spark.SparkContext._
import org.apache.spark

import org.apache.spark.sql.{Row, SparkSession, DataFrame}
import org.apache.spark.ml.feature.{ChiSqSelector, LabeledPoint, StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.stat.Correlation
import shapeless.syntax.std.tuple.productTupleOps

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator


object Main {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)


    val spark = SparkSession.builder()
      .appName("InjuryPrediction")
      .master("local[4]")
      .getOrCreate()

    // Load CSV file into DataFrame
    val data = spark.read.option("header", "true").option("inferSchema", "true").csv("injury_data.csv").filter(row => !row.anyNull)

    // Assemble features into a single vector
    val featureCols = Array("Player_Age", "Player_Weight", "Player_Height", "Previous_Injuries", "Training_Intensity", "Recovery_Time")
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    val assembledData = assembler.transform(data)

    // Split the data into training and testing sets
    val Array(trainData, testData) = assembledData.randomSplit(Array(0.8, 0.2), seed = 12345)

    // Define logistic regression model
    val lr = new LogisticRegression().setLabelCol("Likelihood_of_Injury").setFeaturesCol("features")

    // Define a pipeline
    val pipeline = new Pipeline().setStages(Array(lr))

    // Train the model
    val model = pipeline.fit(trainData)

    // Make predictions on the test data
    val predictions = model.transform(testData)

    // Evaluate the model
    val evaluator = new BinaryClassificationEvaluator().setLabelCol("Likelihood_of_Injury").setRawPredictionCol("rawPrediction")
    val areaUnderROC = evaluator.evaluate(predictions)
    println(s"Area under ROC curve: $areaUnderROC")


  }

}