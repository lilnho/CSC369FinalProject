import breeze.linalg.{sum => breezeSum}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.functions.{col, desc, sum => sparkSum, _}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions.{log => sparkLog, _}
import breeze.numerics.{log => breezeLog}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.PipelineModel

case class Node(attribute: Option[String], children: Map[String, Node], leafValue: Option[Double]) {
  override def toString: String = {
    def printTree(node: Node, prefix: String): String = {
      node.attribute match {
        case Some(attr) =>
          val childStrings = node.children.map { case (attrVal, childNode) =>
            s"$prefix|   $attr <= $attrVal ${printTree(childNode, prefix + "|   ")}"
          }
          childStrings.mkString("\n")
        case None =>
          s"Predict: ${node.leafValue.get}"
      }
    }
    printTree(this, "")
  }

  def predict(row: org.apache.spark.sql.Row): Double = {
    attribute match {
      case Some(attr) =>
        val attrIndex = row.fieldIndex(attr)
        val attrValue = row.get(attrIndex).toString
        children.get(attrValue) match {
          case Some(childNode) => childNode.predict(row)
          case None => leafValue.get // default to the leaf value if the attribute value is not found in the children
        }
      case None => leafValue.get
    }
  }

  def transform(data: DataFrame, decisionTree: Node): DataFrame = {
    import data.sparkSession.implicits._

    val predictions = data.map(row => decisionTree.predict(row))
    val predictionsDF = predictions.toDF("prediction")

    // Join the original DataFrame with the predictions DataFrame
    val dataWithIndex = data.withColumn("index", monotonically_increasing_id())
    val predictionsWithIndex = predictionsDF.withColumn("index", monotonically_increasing_id())
    val dataWithPredictions = dataWithIndex.join(predictionsWithIndex, "index").drop("index")

    dataWithPredictions
  }

}
object DecisionTreeScratch {
  def calculateEntropy(data: DataFrame, labelCol: String): Double = {
    val totalRows = data.count()
    val labelCounts = data.groupBy(labelCol).count()
    val labelProbabilities = labelCounts.withColumn("probability", col("count") / totalRows)
    val entropy = -labelProbabilities.select("probability").collect().map { p =>
      val prob = p.getDouble(0)
      prob * math.log(prob) / math.log(2)
    }.sum
    entropy
  }

  def calculateInformationGain(data: DataFrame, featureCol: String, labelCol: String): Double = {
    val totalEntropy = calculateEntropy(data, labelCol)
    val featureCounts = data.groupBy(featureCol).count()
    val weightedEntropy = featureCounts
      .withColumn("probability", col("count") / sparkSum("count").over(Window.partitionBy()))
      .withColumn("entropy", -col("probability") * sparkLog(col("probability")))
      .agg(sparkSum("entropy").as("weightedEntropy"))
      .first()
      .getDouble(0)
    totalEntropy - weightedEntropy
  }
  def findBestAttribute(data: DataFrame, featureColumns: Array[String], targetColumn: String): String = {
    featureColumns.map { feature =>
      (feature, calculateInformationGain(data, feature, targetColumn))
    }.maxBy(_._2)._1
  }

  def splitData(data: DataFrame, attribute: String, value: Any): DataFrame = {
    data.filter(col(attribute) === value)
  }

  // Check if a subset of data is pure (contains instances of only one class)
  def isPure(data: DataFrame, target: String): Boolean = {
    data.select(target).distinct().count() == 1
  }

  // Add a child node to a parent node in the decision tree structure
  def addChild(parentNode: Node, attributeValue: String, childNode: Node): Unit = {
    val updatedChildren = parentNode.children + (attributeValue -> childNode)
    parentNode.copy(children = updatedChildren)
  }

  // Determine the class label for a leaf node based on the majority class in the subset
  def determineLeafValue(data: DataFrame): Double = {
    val values = data.collect().map(row => {
      if (row.get(0).isInstanceOf[Integer]) {
        row.getInt(0).toDouble
      } else {
        row.getDouble(0)
      }
    })
    values.sum / values.length
  }

  def buildDecisionTree(data: DataFrame, featureColumns: Array[String], targetColumn: String, maxDepth: Int): Node = {
    // Check if the data is pure or if the maximum depth is reached
    if (isPure(data, targetColumn) || maxDepth == 0) {
      // Create a leaf node with the majority class label as the leaf value
      val leafValue = determineLeafValue(data)
      Node(None, Map.empty, Some(leafValue))
    } else {
      // Select the best attribute for splitting
      val bestAttribute = findBestAttribute(data, featureColumns, targetColumn)

      // Determine the unique values of the best attribute
      val attributeValues = data.select(bestAttribute).distinct().collect().map(_.get(0))

      // Create a Node for the decision tree and immediately create child nodes for all unique values of the best attribute
      val children = attributeValues.map { value =>
        val subset = splitData(data, bestAttribute, value)
        val remainingFeatureColumns = featureColumns.filterNot(_ == bestAttribute)
        value.toString -> buildDecisionTree(subset, remainingFeatureColumns, targetColumn, maxDepth - 1)
      }.toMap

      Node(Some(bestAttribute), children, None)
    }
  }

  def calculateAccuracy(predictions: DataFrame): Double = {
    val total = predictions.count()
    val correct = predictions.filter(col("prediction") === col("label")).count()
    correct.toDouble / total.toDouble
  }

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

    // split the dataset
    val Array(trainingData, testData) = injuryData.randomSplit(Array(0.7, 0.3))

    // Calculate entropy for the training and test data
    val trainingEntropy = calculateEntropy(trainingData, targetColumn)
    val testEntropy = calculateEntropy(testData, targetColumn)


    // Calculate information gain for each feature in both training and test data
    val infoGainResults = featureColumns.map { feature =>
      val trainingInformationGain = calculateInformationGain(trainingData, feature, targetColumn)
      val testInformationGain = calculateInformationGain(testData, feature, targetColumn)
      (feature, trainingInformationGain, testInformationGain)
    }

    // Print the results
    infoGainResults.foreach { case (feature, trainingInfoGain, testInfoGain) =>
      println(s"Information Gain for $feature in training data: $trainingInfoGain")
      println(s"Information Gain for $feature in test data: $testInfoGain")
    }

    // Select the best attribute
    val bestAttribute = findBestAttribute(injuryData, featureColumns, targetColumn)

    println(s"Entropy of the training data: $trainingEntropy")
    println(s"Entropy of the test data: $testEntropy")
    println(s"The best attribute for splitting is: $bestAttribute")

    // Build the decision tree
    val maxDepth = 7
    //val decisionTree = buildDecisionTree(injuryData, featureColumns, targetColumn, maxDepth)

    // Print the decision tree
//    println("Decision Tree:")
//    println(decisionTree.toString())
//
//    // Assuming you have already trained your decision tree model and made predictions on the test dataset
//    val predictions = decisionTree.transform(testData, decisionTree)

//    val accuracy = calculateAccuracy(predictions)
//    println(s"Accuracy: $accuracy")

    // Stop SparkSession
    spark.stop()
  }
}
