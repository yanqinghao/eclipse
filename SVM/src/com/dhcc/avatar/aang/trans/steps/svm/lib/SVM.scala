package com.dhcc.avatar.aang.trans.steps.svm.lib

import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.SQLContext
import org.apache.spark.mllib.regression.LabeledPoint
import scala.collection.immutable.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.classification.SVMWithSGD

class SVM(@transient val sc: SparkContext, val initMode: String, val numClusters: Int, val numIterations: Int) extends Serializable {

}

object SVM {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("SVM").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    val path = "E:\\桌面\\svmdata\\train.csv"
    val path1 = "E:\\桌面\\svmdata\\predict.csv"
    val fileData = sc.textFile(path)
    val rddRow = fileData.map(_.split(",")).map(f => Row(f(0).toDouble, f(1).toDouble, f(2).toDouble))
    val schemaString = "c1 c2 label"
    val schema = StructType(schemaString.split(" ").map(a => StructField(a, DoubleType, true)))
    val fileDataFrame = sqlContext.createDataFrame(rddRow, schema)
    val training = fileDataFrame.rdd.map(r => LabeledPoint(r.getAs[Double]("label"), Vectors.dense(r.get(0).asInstanceOf[Double], r.get(1).asInstanceOf[Double])))
    //val testing = MLUtils.loadLibSVMFile(sc, path1)

    val fileData1 = sc.textFile(path1)
    val rddRow1 = fileData1.map(_.split(",")).map(f => Row(f(0).toDouble, f(1).toDouble, f(2).toDouble))
    val schemaString1 = "c1 c2 label"
    val schema1 = StructType(schemaString1.split(" ").map(a => StructField(a, DoubleType, true)))
    val fileDataFrame1 = sqlContext.createDataFrame(rddRow1, schema1)
    val testing = fileDataFrame1.rdd.map(r => LabeledPoint(r.getAs[Double]("label"), Vectors.dense(r.get(0).asInstanceOf[Double], r.get(1).asInstanceOf[Double])))
    //val testing = training
    val numIterations = 1000
    val stepSize = 1
    val miniBatchFraction = 1.0
    val model = SVMWithSGD.train(training, numIterations)
    val prediction = model.predict(testing.map(_.features))
    val predictionAndLabel = prediction.zip(testing.map(_.label))
    val result = sqlContext.createDataFrame(predictionAndLabel).toDF("pre", "real")
    result.show()
    result.rdd.saveAsTextFile("E:\\桌面\\svmdata\\result")
  }
}