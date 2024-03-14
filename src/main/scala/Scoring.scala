
import scala.language.postfixOps

object Scoring {
  // Base score for the class rank
  def scoreHand(hand: String): Double = {
    val values = hand.split(",").map(_.toInt)
    val rankSuitPairs = values.dropRight(1).grouped(2).map(pair => (pair(0), pair(1)))
    val classValue = values.last

    // Extract lists of ranks and suits
    val ranks = rankSuitPairs.map(_._2).toList
    val suits = rankSuitPairs.map(_._1).toList

    def getScore(cRank: Int): Double = {
      cRank match {
        case 9 => 10000.0 // Royal Flush
        case 8 => 9000.0 // Straight Flush
        case 7 => 8000.0 // Four of a Kind
        case 6 => 7000.0 // Full House
        case 5 => 6000.0 // Flush
        case 4 => 5000.0 // Straight
        case 3 => 4000.0 // Three of a Kind
        case 2 => 3000.0 // Two Pairs
        case 1 => 2000.0 // One Pair
        case _ => 1000.0 // High Card or No hand
      }
    }

    // Additional score based on the ranks
    //using index so that lower indexes are weighted less


    def isAlmostFlush(suits: List[Int]): Boolean = {
      suits.groupBy(identity).values.exists(_.size == 4)
    }


    // Helper method to get the length of the longest consecutive sequence in the hand
    //check if there is a potential straight
    def longestConsecutiveSequence(ranks: List[Int]): Int = {
      val sortedRanks = ranks.sorted
      sortedRanks.foldLeft((0, 0)) { case ((currentStreak, longestStreak), rank) =>
        if (currentStreak == 0 || rank - 1 == sortedRanks(sortedRanks.indexOf(rank) - 1))
          (currentStreak + 1, Math.max(currentStreak + 1, longestStreak))
        else
          (1, longestStreak)
      }._2
    }
    val rankScores = ranks.map(rank => if (rank == 1) 14 else 14 - rank).sum
    val classScore = getScore(classValue)
    val maybeFlush = if (isAlmostFlush(suits)) 25 else 0
    val totalScore = classScore + (rankScores * classScore * 0.002) + maybeFlush + (longestConsecutiveSequence(ranks) * 4)
    return totalScore
  }
def main(args: Array[String]): Unit = {
    val input = "1,13,2,6,1,6,2,11,3,5,1"
    print(scoreHand(input))
  }

}