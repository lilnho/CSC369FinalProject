import org.apache.commons.math3.distribution.NormalDistribution
import org.apache.commons.math3.stat.descriptive.MultivariateSummaryStatistics
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.stat.distribution._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd._
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary
object NaiveBayes {
  // input: rdd you want the mean values of
  // (Age, Weight, Height, PreviousInjuries, Intensity, RecoveryTime) ---- excludes likelihood
  def calcMean(rdd: RDD[(Double, Double, Double, Double, Double, Double, Double)]): (Double, Double, Double, Double, Double, Double) = {

    val sums = rdd.reduce{
      case(x, y) => (x._1 + y._1, x._2 + y._2, x._3 + y._3, x._4 + y._4, x._5 + y._5, x._6 + y._6, x._7 + y._7)
    }
    val counts = rdd.count().toDouble

    ((sums._1 / counts).toInt,
      sums._2 / counts,
      sums._3 / counts,
      (sums._4 / counts).toInt,
      sums._5 / counts,
      (sums._6 / counts).toInt)
  }

  // returns tuple of standard deviation given an RDD of data
  def calcStdDev(rdd: RDD[(Double, Double, Double, Double, Double, Double, Double)], means: (Double, Double, Double, Double, Double, Double)): (Double, Double, Double, Double, Double, Double) = {
    val squares = rdd.map{
      case (age, w, h, prevInjury, intensity, recovery, likelihood) =>
        val x1 = age - means._1
        val x2 = w - means._2
        val x3 = h - means._3
        val x4 = prevInjury - means._4
        val x5 = intensity - means._5
        val x6 = recovery - means._6
        (x1*x1, x2*x2, x3*x3, x4*x4, x5*x5, x6*x6)
    }

    val summed = squares.reduce {
      case((s1, s2, s3, s4, s5, s6), (y1, y2, y3, y4, y5, y6)) =>
        (s1+y1, s2+y2, s3+y3, s4+y4, s5+y5, s6+y6)
    }

    val count = rdd.count()

    val stdDev1 = math.sqrt(summed._1 / count)
    val stdDev2 = math.sqrt(summed._2 / count)
    val stdDev3 = math.sqrt(summed._3 / count)
    val stdDev4 = math.sqrt(summed._4 / count)
    val stdDev5 = math.sqrt(summed._5 / count)
    val stdDev6 = math.sqrt(summed._6 / count)
    (stdDev1, stdDev2, stdDev3, stdDev4, stdDev5, stdDev6)
  }

    // Computes the probability density of x in a univariate Gaussian distribution
    def gaussianProbability(x: Double, mean: Double, variance: Double): Double = {
      val numerator = math.exp(-math.pow(x - mean, 2) / (2 * variance))
      val denominator = math.sqrt(2 * math.Pi * variance)
      numerator / denominator
    }
    // Your Naive Bayes prediction function
    def predictInjury(features: Array[Double],
                      injuredMeans: Array[Double], injuredVariances: Array[Double],
                      notInjuredMeans: Array[Double], notInjuredVariances: Array[Double],
                      priorInjured: Double, priorNotInjured: Double): Double = {
      val injuredProb = math.log(priorInjured) + features.zip(injuredMeans).zip(injuredVariances).map {
        case ((x, mean), variance) => math.log(gaussianProbability(x, mean, variance))
      }.sum

      val notInjuredProb = math.log(priorNotInjured) + features.zip(notInjuredMeans).zip(notInjuredVariances).map {
        case ((x, mean), variance) => math.log(gaussianProbability(x, mean, variance))
      }.sum

      if (injuredProb > notInjuredProb) 1.0 else 0.0
    }


  def main(args: Array[String]): Unit = {
    print("hello")

    val conf = new SparkConf().setAppName("NaiveBayesInjuryPrediction").setMaster("local[4]")
    val sc = new SparkContext(conf)
    // Load the data from a CSV file

    //Age, Weight, Height, Previous_Injuries, Training_Intensity, Recovery_Time, Likelihood_of_Injury
    val injuryData = sc.textFile("injury_data.csv")
    // Split the data into training and test sets (80% training, 20% testing)
    val seed = 12345L // Seed for reproducibility
    val Array(trainingData, testData) = injuryData.randomSplit(Array(0.9, 0.1), seed)

    // Cache the training data for efficiency
    trainingData.cache()

    val trainData = trainingData.filter(line => line.split(",").length == 7).map{line =>
      val parts = line.split(",")
      val age = parts(0).trim.toDouble
      val w = parts(1).trim.toDouble
      val h = parts(2).trim.toDouble
      val prev = parts(3).trim.toDouble
      val intensity = parts(4).trim.toDouble
      val recover = parts(5).trim.toDouble
      val likely = parts(6).trim.toDouble
      (age, w, h, prev, intensity, recover, likely)
    }

    //split data based on likely or not likely
    val dataLikely = trainData.filter(_._7.toInt == 1).persist()
    val dataUnlikely = trainData.filter(_._7.toInt == 0).persist()

    val likelyMeans = calcMean(dataLikely)
    val unlikelyMeans = calcMean(dataUnlikely)

    val likelyVariances = calcStdDev(dataLikely, likelyMeans).productIterator.map(_.asInstanceOf[Double]).toArray
    val unlikelyVariances = calcStdDev(dataUnlikely, unlikelyMeans).productIterator.map(_.asInstanceOf[Double]).toArray

    val arrLikelyMeans = likelyMeans.productIterator.map(_.asInstanceOf[Double]).toArray
    val arrUnlikelyMeans = unlikelyMeans.productIterator.map(_.asInstanceOf[Double]).toArray

    val likelyCount = dataLikely.count()
    val unlikelyCount = dataUnlikely.count()
    val totalCount = likelyCount + unlikelyCount
    val likelyGuess = likelyCount / totalCount
    val unlikelyGuess = unlikelyCount / totalCount


    // 2. Apply the predictInjury function to each line
    val processedTestData = testData.map(line => {
      val fields = line.split(",")
      val features = fields.init.map(_.toDouble) // Exclude the last field
      val actualLabel = fields.last.toDouble // Get the actual label

      // Call predictInjury with the features and other parameters
      val predictedLabel = predictInjury(features, arrLikelyMeans, likelyVariances, arrUnlikelyMeans, unlikelyVariances, likelyGuess, unlikelyGuess)

      // Return a tuple containing the original line and the predicted label
      (line, predictedLabel, actualLabel)
    })

    // 4. Compare the new dataset with the original test data to calculate accuracy
    val accuracy = processedTestData.map {
      case (_, predictedLabel, actualLabel) =>
        if (predictedLabel == actualLabel) 1 else 0
    }.mean() * 100 // Convert to percentage



    // Print the accuracy
    println(s"Accuracy: $accuracy%")


  }
}
/*
Split the data into 2 datasets of likely and unlikely
get the normal distribution of each category for both likely and unlikely and find mean and standard deviation for each
make a guess that the person is likely to be injured and initial guess will be 0.5
initial guess for unlikely is also 0.5
the score for likely is
lne(initial guess * P(height | likely) * P(weight | likely))
z-score = (x – μ) / σ
*/

