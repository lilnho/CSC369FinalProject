import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

object InfoGain {
  // Calculate entropy
  def entropy(data: DataFrame, labelCol: String): Double = {
    val totalRows = data.count()
    val labelCounts = data.groupBy(labelCol).count()
    val labelProbabilities = labelCounts.withColumn("probability", col("count") / totalRows)
    val entropy = -labelProbabilities.select("probability").collect().map { p =>
      val prob = p.getDouble(0)
      prob * math.log(prob) / math.log(2)
    }.sum
    entropy
  }

  // Calculate information gain
  def informationGain(data: DataFrame, featureCol: String, labelCol: String): Double = {
    val totalEntropy = entropy(data, labelCol)
    val featureCounts = data.groupBy(featureCol).count()
    val weightedEntropy = featureCounts
      .withColumn("probability", col("count") / sum("count").over(Window.partitionBy()))
      .withColumn("entropy", -col("probability") * log(col("probability")))
      .groupBy()
      .agg(sum("entropy").as("weightedEntropy"))
      .first()
      .getDouble(0)
    totalEntropy - weightedEntropy
  }

  // Find the best attribute based on entropy
  def findAttribute(data: DataFrame, featureCols: Array[String], labelCol: String): (String, Double) = {
    featureCols.map { feature =>
      (feature, informationGain(data, feature, labelCol))
    }.maxBy(_._2)
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val spark = SparkSession.builder()
      .appName("InfoGain")
      .master("local[4]")
      .getOrCreate()

    try {
      val data = spark.read
        .option("header", "false")
        .option("inferSchema", "true")
        .csv("injury_data.csv")

      val numColumns = data.columns.length
      val featureCols = data.columns.dropRight(1)
      val labelCol = data.columns.last

      // Find the best split based on entropy
      val (bestFeature, bestInformationGain) = findAttribute(data, featureCols, labelCol)

      // Map the best feature to original column name
      val columnMapping = Map(
        "_c0" -> "Player_Age",
        "_c1" -> "Player_Weight",
        "_c2" -> "Player_Height",
        "_c3" -> "Previous_Injuries",
        "_c4" -> "Training_Intensity",
        "_c5" -> "Recovery_Time",
        "_c6" -> "Likelihood_of_Injury"
      )
      val bestFeatureMapped = columnMapping(bestFeature)

      println(s"Best feature to split on: $bestFeatureMapped")
      println(s"Information gain: $bestInformationGain")
    } catch {
      case e: Exception => println("Error occurred: " + e.getMessage)
    } finally {
      spark.stop()
    }
  }
}
