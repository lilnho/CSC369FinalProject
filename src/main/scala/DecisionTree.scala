import scala.collection.mutable
import scala.util.Random
case class Node(featureIndex: Int, threshold: Double, leftChild: Option[Node], rightChild: Option[Node], prediction: Option[Int])
class DecisionTree(maxDepth: Int) {
  var rootNode: Option[Node] = None
  def fit(X: Array[Array[Double]], y: Array[Int]): Unit = {
    rootNode = Some(buildTree(X, y, maxDepth))
  }
  def predict(X: Array[Array[Double]]): Array[Int] = {
    require(rootNode.isDefined, "Decision tree has not been trained")
    X.map(predictOne)
  }
  private def predictOne(instance: Array[Double]): Int = {
    var node = rootNode
    while (node.get.prediction.isEmpty) {
      if (instance(node.get.featureIndex) <= node.get.threshold) {
        node = node.get.leftChild
      } else {
        node = node.get.rightChild
      }
    }
    node.get.prediction.get
  }

  // Method to build the decision tree recursively
  private def buildTree(X: Array[Array[Double]], y: Array[Int], depth: Int): Node = {
    val numFeatures = X.head.length
    val numSamples = X.length

    // Base cases
    if (depth == 0 || y.toSet.size == 1) {
      // Leaf node, make prediction
      val prediction = y.groupBy(identity).maxBy(_._2.length)._1
      return Node(-1, -1, None, None, Some(prediction))
    }

    // Find the best split
    var bestGini = Double.MaxValue
    var bestFeatureIndex = -1
    var bestThreshold = -1.0
    for (featureIndex <- 0 until numFeatures) {
      val values = X.map(_(featureIndex)).distinct.sorted
      for (i <- 0 until (values.length - 1)) {
        val threshold = (values(i) + values(i + 1)) / 2.0
        val (left, right) = X.zip(y).partition(x => x._1(featureIndex) <= threshold)
        val gini = giniImpurity(left.map(_._2), right.map(_._2))
        if (gini < bestGini) {
          bestGini = gini
          bestFeatureIndex = featureIndex
          bestThreshold = threshold
        }
      }
    }

    // Split the data based on the best split
    val (left, right) = X.zip(y).partition(x => x._1(bestFeatureIndex) <= bestThreshold)
    val leftX = left.map(_._1)
    val leftY = left.map(_._2)
    val rightX = right.map(_._1)
    val rightY = right.map(_._2)

    // Recursively build left and right subtrees
    val leftChild = if (leftY.isEmpty) None else Some(buildTree(leftX, leftY, depth - 1))
    val rightChild = if (rightY.isEmpty) None else Some(buildTree(rightX, rightY, depth - 1))

    Node(bestFeatureIndex, bestThreshold, leftChild, rightChild, None)
  }

  // Gini impurity calculation
  private def giniImpurity(leftY: Array[Int], rightY: Array[Int]): Double = {
    val totalSize = leftY.length + rightY.length
    val leftGini = if (leftY.isEmpty) 0 else leftY.groupBy(identity).mapValues(_.length.toDouble / leftY.length).values.map(p => p * (1 - p)).sum
    val rightGini = if (rightY.isEmpty) 0 else rightY.groupBy(identity).mapValues(_.length.toDouble / rightY.length).values.map(p => p * (1 - p)).sum
    (leftGini * leftY.length + rightGini * rightY.length) / totalSize
  }

  // Method to visualize the decision tree using Graphviz
  def visualize(): Unit = {
    require(rootNode.isDefined, "Decision tree has not been trained")

    val sb = new mutable.StringBuilder()
    sb.append("digraph DecisionTree {\n")

    def buildGraph(node: Option[Node], nodeId: String): Unit = {
      if (node.isDefined) {
        val featureName = s"Feature_${node.get.featureIndex}"
        sb.append(s"""$nodeId [label="$featureName <= ${node.get.threshold}\\nPrediction: ${node.get.prediction.getOrElse("N/A")}"];\n""")
        if (node.get.leftChild.isDefined) {
          val leftId = nodeId + "_L"
          sb.append(s"$nodeId -> $leftId;\n")
          buildGraph(node.get.leftChild, leftId)
        }
        if (node.get.rightChild.isDefined) {
          val rightId = nodeId + "_R"
          sb.append(s"$nodeId -> $rightId;\n")
          buildGraph(node.get.rightChild, rightId)
        }
      }
    }

    buildGraph(rootNode, "Root")
    sb.append("}")

    println(sb.toString())
  }
}

object DecisionTreeExample {
  def main(args: Array[String]): Unit = {
    // Sample dataset
    val X = Array(
      Array(24, 66.25193286255299, 175.73242883117646, 1, 0.4579289944340279, 5),
      Array(37, 70.99627126832448, 174.58165012331358, 0, 0.2265216260361057, 6),
      Array(32, 80.09378116336106, 186.32961751509828, 0, 0.6139703063252326, 2),
      Array(28, 87.47327055231725, 175.50423961774717, 1, 0.2528581182501112, 4),
      Array(25, 84.6592200795959, 190.1750122908418, 0, 0.5776317543444226, 1)
    )
    val y = Array(0, 1, 1, 1, 1)

    // Create and train the decision tree
    val decisionTree = new DecisionTree(maxDepth = 3)
    decisionTree.fit(X, y)

    // Visualize the decision tree
    decisionTree.visualize()
  }
}
