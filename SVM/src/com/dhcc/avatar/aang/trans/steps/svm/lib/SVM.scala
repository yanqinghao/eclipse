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
import org.apache.spark.mllib.classification.SVMModel
import org.apache.spark.sql.DataFrame
import java.io.File
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.DataFrame
import org.dmg.pmml.True

class SVM(@transient val sc: SparkContext, val numIterations: Int) extends Serializable {

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

  private def toVec(arr: Array[Double], size: Int) = {
    val data = new Array[Double](size - 1)
    for (i <- 0 to size - 2)
      data(i) = arr(i)
    data
  }

  def svmTrain(trainData: DataFrame, path: String): SVMModel = {
    val col = trainData.columns
    val size = col.length
    val rddArr = trainData.rdd.map(toArr(_, size))
    val parsedData = rddArr.map(r => LabeledPoint(r(size - 1), Vectors.dense(toVec(r, size))))
    val model = SVMWithSGD.train(parsedData, numIterations)
    val file = new File(path)
    if (file.exists()) {
      val delFlag = deleteDir(file)
      val dirFlag = file.mkdir()
    }
    model.save(sc, path)
    model
  }

  def svmLoad(path: String): SVMModel = {
    val model: SVMModel = SVMModel.load(sc, path)
    model
  }

  def svmPredict(model: SVMModel, preData: DataFrame, withLabel: Boolean): DataFrame = {
    val res = new Array[DataFrame](1)
    if (withLabel) {
      val col = preData.columns
      val size = col.length
      val rddArr = preData.rdd.map(toArr(_, size))
      val parsedData = rddArr.map(r => LabeledPoint(r(size - 1), Vectors.dense(toVec(r, size))))
      val index = model.predict(parsedData.map(_.features))
      val predictionAndLabel = index.zip(parsedData.map(_.label))
      val result = sqlContext.createDataFrame(predictionAndLabel).toDF("pre", "real")
      res(0) = result
    } else {
      val col = preData.columns
      val size = col.length
      val rddArr = preData.rdd.map(toArr(_, size))
      val parsedData = rddArr.map(r => Vectors.dense(r))
      val index = model.predict(parsedData)
      val pre = index.map(r => Tuple1(r))
      val result = sqlContext.createDataFrame(pre).toDF("pre")
      res(0) = result
    }
    res(0)
  }

}

object SVM {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("SVM").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    val trainpath = "E:\\桌面\\svmdata\\train.csv"
    val prepathlab = "E:\\桌面\\svmdata\\predict.csv"
    val prepath = "E:\\桌面\\svmdata\\predict1.csv"
    val trainData = sc.textFile(trainpath)
    val rddRow = trainData.map(_.split(",")).map(f => Row(f(0).toDouble, f(1).toDouble, f(2).toDouble))
    val schemaString = "c1 c2 label"
    val schema = StructType(schemaString.split(" ").map(a => StructField(a, DoubleType, true)))
    val trainDF = sqlContext.createDataFrame(rddRow, schema)

    val preDatalab = sc.textFile(prepathlab)
    val rddRow1 = preDatalab.map(_.split(",")).map(f => Row(f(0).toDouble, f(1).toDouble, f(2).toDouble))
    val schemaString1 = "c1 c2 label"
    val schema1 = StructType(schemaString1.split(" ").map(a => StructField(a, DoubleType, true)))
    val preDFlab = sqlContext.createDataFrame(rddRow1, schema1)

    val preData = sc.textFile(prepath)
    val rddRow2 = preData.map(_.split(",")).map(f => Row(f(0).toDouble, f(1).toDouble))
    val schemaString2 = "c1 c2"
    val schema2 = StructType(schemaString2.split(" ").map(a => StructField(a, DoubleType, true)))
    val preDF = sqlContext.createDataFrame(rddRow2, schema2)

    val withLabel = true
    val numIterations = 1000
    val modelPath = "E://svmMdl//"
    val svmIns = new SVM(sc, numIterations)
    val model = svmIns.svmTrain(trainDF, modelPath)
    val result = svmIns.svmPredict(model, preDFlab, withLabel)
    val result2 = svmIns.svmPredict(model, preDF, false)
    val model1 = svmIns.svmLoad(modelPath)
    val result1 = svmIns.svmPredict(model1, preDFlab, withLabel)
    val result3 = svmIns.svmPredict(model1, preDF, false)
    result.show()
    result1.show()
    result2.show()
    result3.show()
  }
}