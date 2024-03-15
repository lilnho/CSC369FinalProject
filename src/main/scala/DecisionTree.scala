import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.util.MLWriter

object DecisionTree {
  def main(args: Array[String]): Unit = {
    // Create SparkSession
    val spark = SparkSession.builder()
      .appName("DecisionTree")
      .master("local[*]")
      .getOrCreate()

    // Define schema for the CSV file
    val schema = StructType(Seq(
      StructField("Player_Age", IntegerType, nullable = true),
      StructField("Player_Weight", DoubleType, nullable = true),
      StructField("Player_Height", DoubleType, nullable = true),
      StructField("Previous_Injuries", IntegerType, nullable = true),
      StructField("Training_Intensity", DoubleType, nullable = true),
      StructField("Recovery_Time", IntegerType, nullable = true),
      StructField("Likelihood_of_Injury", IntegerType, nullable = true)
    ))

    // Read the CSV file without a header row
    val injuryData = spark.read.schema(schema).option("header", "false").csv("injury_data.csv")

    // Define feature columns and target column
    val featureColumns = Array("Player_Age", "Player_Weight", "Player_Height", "Previous_Injuries", "Training_Intensity", "Recovery_Time")
    val targetColumn = "Likelihood_of_Injury"

    // Convert target column to categorical
    val indexer = new StringIndexer()
      .setInputCol(targetColumn)
      .setOutputCol("label")

    // Assemble features into a single vector
    val assembler = new VectorAssembler()
      .setInputCols(featureColumns)
      .setOutputCol("features")

    // Create a DecisionTreeClassifier
    val decisionTree = new DecisionTreeClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setImpurity("entropy")

    // Create a Pipeline
    val pipeline = new Pipeline()
      .setStages(Array(indexer, assembler, decisionTree))

    // Split the data into training and test sets
    val Array(trainingData, testData) = injuryData.randomSplit(Array(0.7, 0.3))

    // Train the model on the training data
    val model = pipeline.fit(trainingData)

    // Make predictions on the test data
    val predictions = model.transform(testData)

    // Select (prediction, true label) and compute test error
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
    val precision = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
    val recall = evaluator.setMetricName("weightedRecall").evaluate(predictions)
    val f1 = evaluator.setMetricName("f1").evaluate(predictions)

    println(s"Accuracy: $accuracy")
    println(s"Precision: $precision")
    println(s"Recall: $recall")
    println(s"F1 Score: $f1")

    // Extract the decision tree model
    val decisionTreeModel = model.stages.last.asInstanceOf[DecisionTreeClassificationModel]

    // Get the debug string
    val debugString = decisionTreeModel.toDebugString

    // Replace feature indices with feature names
    val featureNames = assembler.getInputCols
    val replacedDebugString = featureNames.indices.foldLeft(debugString) { (str, i) =>
      str.replaceAll(s"feature $i", featureNames(i))
    }

    // Print out the decision tree
    println("Decision Tree Model:")
    println(replacedDebugString)

    // Stop SparkSession
    spark.stop()
  }
}