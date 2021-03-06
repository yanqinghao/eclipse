package com.dhcc.avatar.aang.trans.appsteps.Kmeans.lib

import scala.collection.immutable.Vector
import org.apache.spark.mllib.clustering.KMeansModel
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.Row
import org.apache.spark.sql.DataFrame
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.rdd.RDD
import scala.collection.immutable.Vector
import java.util.Vector
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.DataFrame
import scala.io.Source
import org.apache.spark.annotation.Private
import org.apache.hadoop.yarn.webapp.hamlet.HamletSpec.Dir
import java.io.File
import org.apache.derby.impl.sql.compile.DeleteNode

/**
 * *
 * k-means clustering aims to partition n observations into k clusters in which each observation belongs
 * to the cluster with the nearest mean, serving as a prototype of the cluster.
 * input:
 * val testData = Array((1.0, 1.1), (2.0, 1.0), (0.9, 1.0), (3.7, 4.0), (3.9, 3.9),
 * (3.6, 4.1), (10.0, 10.0), (2.9, 1.0), (10.1, 9.9), (3.9, 1.0))
 * val initMode = "K-means||" //val initMode = "random"
 * val numClusters = 3
 * val numIterations = 20
 * val modelPath = "E://KmeansMdl//"
 * result:
 * +----+----+-----+
 * |  v0|  v1|label|
 * +----+----+-----+
 * | 1.0| 1.1|  0.0|
 * | 2.0| 1.0|  0.0|
 * | 0.9| 1.0|  0.0|
 * | 3.7| 4.0|  2.0|
 * | 3.9| 3.9|  2.0|
 * | 3.6| 4.1|  2.0|
 * |10.0|10.0|  1.0|
 * | 2.9| 1.0|  0.0|
 * |10.1| 9.9|  1.0|
 * | 3.9| 1.0|  0.0|
 * +----+----+-----+
 *
 * cluster center:
 * +-----------------+----+
 * |               v0|  v1|
 * +-----------------+----+
 * |             2.14|1.02|
 * |            10.05|9.95|
 * |3.733333333333333| 4.0|
 * +-----------------+----+
 *
 *
 */

class KmeansLib(@transient val sc: SparkContext) extends Serializable {

  val sqlContext = new SQLContext(sc)

  private def deleteDir(dir: File): Boolean = {
    if (dir.isDirectory()) {
      val children = dir.list()
      for (i <- 0 until children.length) {
        val success = deleteDir(new File(dir, children(i)))
        if (!success)
          return false
      }
    }
    return dir.delete()
  }

  private def toArr(dt: Row, size: Int) = {
    val data = new Array[Double](size)
    for (i <- 0 to size - 1)
      data(i) = dt.getAs[Double](i)
    data
  }

  private def arrToRow(input: Array[Double]) = {
    var dataSeq = Seq[Any]()
    for (i <- 0 to input.length - 1) {
      dataSeq = dataSeq.:+(input(i))
    }
    Row.fromSeq(dataSeq)
  }

  private def getOutputSchema(colNum: Int): StructType = {
    val fieldArray = new Array[StructField](colNum)
    for (i <- 0 to colNum - 1) {
      fieldArray(i) = StructField("v" + i.toString(), DoubleType, true);
    }
    new StructType(fieldArray)
  }

  def kmTrain(trainData: DataFrame, path: String, initMode: String, numClusters: Int, numIterations: Int): KMeansModel = {
    val col = trainData.columns
    val size = col.length
    val rddArr = trainData.rdd.map(toArr(_, size))
    val parsedData = rddArr.map(Vectors.dense(_))
    val clusters = KMeans.train(parsedData, numClusters, numIterations, 1, initMode)
    clusters.clusterCenters.foreach(println)
    val file = new File(path)
    if (file.exists()) {
      val delFlag = deleteDir(file)
      val dirFlag = file.mkdir()
    }
    clusters.save(sc, path)
    clusters
  }

  def kmCenter(clusters: KMeansModel): DataFrame = {
    val centerVec = clusters.clusterCenters
    val centerArr = centerVec.map(_.toArray)
    val arrRow = centerArr.map(arrToRow)
    val rddRow = sc.parallelize(arrRow)
    val schemaOutput = getOutputSchema(centerArr(0).length)
    val centerDF = sqlContext.createDataFrame(rddRow, schemaOutput)
    centerDF
  }

  def kmLoad(path: String): KMeansModel = {
    val clusters: KMeansModel = KMeansModel.load(sc, path)
    clusters
  }

  def kmPredict(clusters: KMeansModel, preData: DataFrame): DataFrame = {
    val col = preData.columns
    val size = col.length
    val rddArr = preData.rdd.map(toArr(_, size))
    val parsedData = rddArr.map(Vectors.dense(_))
    val index = clusters.predict(parsedData)
    val predictZip = rddArr.zip(index)
    val rdd = predictZip.map(TpToRow(_))
    val schemaOutput = getDFOutputSchema(size)
    val predictZipDF = sqlContext.createDataFrame(rdd, schemaOutput)
    predictZipDF
  }

  private def TpToRow(input: (Array[Double], Int)) = {
    var dataSeq = Seq[Any]()
    for (i <- 0 to input._1.length - 1) {
      dataSeq = dataSeq.:+(input._1(i))
    }
    dataSeq = dataSeq.:+(input._2.toDouble)
    Row.fromSeq(dataSeq)
  }

  private def getDFOutputSchema(col: Int): StructType = {
    val fieldArray = new Array[StructField](col + 1)
    for (i <- 0 to col - 1) {
      fieldArray(i) = StructField("v" + i.toString(), DoubleType, true);
    }
    fieldArray(col) = StructField("label", DoubleType, true);
    new StructType(fieldArray)
  }

}

object KmeansLib {
  def main(args: Array[String]) {
    val conf = new SparkConf().setMaster("local[2]").setAppName("spark")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    //val fileData = sc.textFile("E:\\桌面\\K-means聚类数据\\ttt.txt")
    val testData = Array((1.0, 1.1), (2.0, 1.0), (0.9, 1.0), (3.7, 4.0), (3.9, 3.9),
      (3.6, 4.1), (10.0, 10.0), (2.9, 1.0), (10.1, 9.9), (3.9, 1.0))
    val rddArr = sc.parallelize(testData)
    val rddRow = rddArr.map(f => Row(f._1, f._2))
    val schemaString = "c1 c2"
    val schema = StructType(schemaString.split(" ").map(a => StructField(a, DoubleType, true)))
    val fileDataFrame = sqlContext.createDataFrame(rddRow, schema)
    fileDataFrame.show()
    val initMode = "K-means||" //val initMode = "random"
    val numClusters = 3
    val numIterations = 20
    val modelPath = "E://KmeansMdl//"
    val km = new KmeansLib(sc)
    val clusters = km.kmTrain(fileDataFrame, modelPath, initMode, numClusters, numIterations)
    val clustersLd = km.kmLoad(modelPath)
    val result = km.kmPredict(clusters, fileDataFrame)
    val resultLd = km.kmPredict(clusters, fileDataFrame)
    val centers = km.kmCenter(clusters)
    val centersLd = km.kmCenter(clusters)
    result.show()
    centers.show()
    resultLd.show()
    centersLd.show()
  }
}
