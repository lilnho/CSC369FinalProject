
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession

import scala.language.postfixOps

object Probability {
  // Base score for the class rank
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val conf = new SparkConf().setAppName("Lab6").setMaster("local")
    val sc = new SparkContext(conf)
    val spark = SparkSession.builder()
      .appName("Poker Hand Classification")
      .master("local[4]")
      .getOrCreate()

    // load data
    val trainingFile = sc.textFile("poker-hand-training-true.data")
      .map(line => line.split(","))
      .map(fields => (fields.take(10).map(_.toInt), fields(10).toInt))

    // Group by class and count occurrences
    val classCounts = trainingFile.map { case (_, classValue) => (classValue, 1) }
      .reduceByKey(_ + _)
      .sortByKey()

    // Total number of hands in the dataset
    val totalHands = trainingFile.count()

    // Calculate the probability of each class and print
    classCounts.collect().foreach { row =>
      val classValue = row._1
      val count = row._2
      val probability = count.toDouble / totalHands
      println(s"Class $classValue Probability: $probability")
    }

    spark.stop()
  }

}