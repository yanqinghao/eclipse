package com.dhcc.avatar.aang.trans.steps.dbscan.lib

import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.sql.{ DataFrame, SQLContext }
import org.apache.spark.sql.functions.{ lit, udf }
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import scala.annotation.tailrec
import scala.math.{ pow, sqrt }
import scala.reflect.internal.util.TableDef.Column

/**
 * Created by leo on 17-6-5.
 * DBSCAN: Density based Spatial Clustering of Applications with Noise
 * Input: DataFrame
 * +----+----+
 * |   X|   Y|
 * +----+----+
 * | 1.0| 1.1|
 * | 2.0| 1.0|
 * | 0.9| 1.0|
 * | 3.7| 4.0|
 * | 3.9| 3.9|
 * | 3.6| 4.1|
 * |10.0|10.0|
 * | 2.9| 1.0|
 * |10.1| 9.9|
 * | 3.9| 1.0|
 * +----+----+
 * Output: DataFrame
 * +----+----+---+----------+----+------------------+
 * |   X|   Y| ID|classified|seed|              dist|
 * +----+----+---+----------+----+------------------+
 * | 1.0| 1.1|  0|         0|   0|12.658988901172163|
 * | 2.0| 1.0|  1|         0|   0| 12.03411816461846|
 * | 0.9| 1.0|  2|         0|   0|12.800390619039717|
 * | 3.7| 4.0|  3|         3|   0| 8.704596486914255|
 * | 3.9| 3.9|  4|         3|   0| 8.627861844049196|
 * | 3.6| 4.1|  5|         3|   0| 8.711486669908874|
 * |10.0|10.0|  6|         6|   0| 0.141421356237309|
 * | 2.9| 1.0|  7|         0|   0|11.447707194019246|
 * |10.1| 9.9|  8|         6|   0|               0.0|
 * | 3.9| 1.0|  9|         0|   0|10.846658471621572|
 * +----+----+---+----------+----+------------------+
 * Parameters:
 *   eps:
 *   minPoints:
 */
case class Node(ID: Int, X: Double, Y: Double, classified: Int)
class DBSCAN(spc: SparkContext, coll: DataFrame, eps: Double, minPoints: Int) extends Serializable {
  @transient val sc = spc
  val Noise = -1
  val Unclassified = -10
  val rowNum = coll.count()
  val sqlContext = new SQLContext(sc)
  import sqlContext.implicits._
  def run(): DataFrame = {
    val schemaString = "X Y ID"
    val id = Range(0, rowNum.toInt)
    val inputData = sc.parallelize(id)
    val schema = StructType(schemaString.split(" ").map(str => if (str == "ID") StructField(str, IntegerType, true) else StructField(str, DoubleType, true)))
    val collRdd = coll.rdd.zipWithIndex().map(a => Row(a._1(0), a._1(1), a._2.toInt))
    val collAddID = sqlContext.createDataFrame(collRdd, schema)
    val marked = collAddID.withColumn("classified", lit(Unclassified)).withColumn("seed", lit(0))
    markCluster(marked)
  }

  @tailrec
  private def markCluster(points: DataFrame): DataFrame = {
    val untested = points.filter($"classified" < Noise).limit(1)
    if (untested.count() < 1) {
      points
    } else {
      val curNode = untested.as[Node].first()
      val markCurrent = udf((id: Int, seed: Int) => if (id == curNode.ID) 1 else seed)
      val markCurNode = points.withColumn("seed", markCurrent($"ID", $"seed"))
      val newCluster = extendNeighbours(markCurNode)
      markCluster(newCluster)
    }
  }

  def markNeighbours(points: DataFrame, node: Node): DataFrame = {
    val dist = udf((x: Double, y: Double, x0: Double, y0: Double) => (sqrt(pow(x - x0, 2) + pow(y - y0, 2))))
    val withDist = points.withColumn("dist", dist($"X", $"Y", lit(node.X), lit(node.Y)))
    val setCluster = udf((classified: Int, dist: Double) =>
      {
        if (dist < eps && node.classified >= 0) node.classified
        else if (dist < eps && node.classified < 0) node.ID
        else classified
      })
    val setNoise = udf((classified: Int, dist: Double) => if (dist < eps) Noise else classified)
    val isNoise = withDist.filter($"dist" < eps).filter($"dist" >= 0).count < minPoints
    val setSeed = udf((id: Int, dist: Double, seed: Int, classified: Int) =>
      if (classified < 0 && dist < eps && id != node.ID) 1
      else if (id == node.ID) 0
      else seed)
    val markSeed = if (isNoise) withDist else withDist.withColumn("seed", setSeed($"ID", $"dist", $"seed", $"classified"))
    val new_classified = markSeed.withColumn("classified",
      if (isNoise) setNoise($"classified", $"dist") else setCluster($"classified", $"dist"))
    new_classified
  }

  @tailrec
  private def extendNeighbours(points: DataFrame): DataFrame = {
    val dataLen = points.filter($"seed" > 0).count()
    if (dataLen == 0) {
      points
    } else {
      val firstSeed = points.filter($"seed" > 0).limit(1).as[Node].first()
      val firstMarked = markNeighbours(points, firstSeed)
      firstMarked.cache()
      extendNeighbours(firstMarked)
    }
  }
}

object DBSCAN {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("DBSCAN").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    /*val testData = Seq((0, 1.0, 1.1), (1, 2.0, 1.0), (2, 0.9, 1.0), (3, 3.7, 4.0), (4, 3.9, 3.9),
      (5, 3.6, 4.1), (6, 10.0, 10.0), (7, 2.9, 1.0), (8, 10.1, 9.9), (9, 3.9, 1.0))*/
    val testData = Seq((1.0, 1.1), (2.0, 1.0), (0.9, 1.0), (3.7, 4.0), (3.9, 3.9),
      (3.6, 4.1), (10.0, 10.0), (2.9, 1.0), (10.1, 9.9), (3.9, 1.0))
    val eps = 1.5
    val minPoints = 2
    val inp = sqlContext.createDataFrame(testData).toDF("X", "Y")
    val dbscan = new DBSCAN(sc, inp, eps, minPoints)
    val res = dbscan.run()
    println("Final result:")
    res.show()
  }
}