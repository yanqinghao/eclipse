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

  def svmPredict(model: SVMModel, preData: DataFrame): DataFrame = {
    val col = preData.columns
    val size = col.length
    val rddArr = preData.rdd.map(toArr(_, size))
    val parsedData = rddArr.map(r => LabeledPoint(r(size - 1), Vectors.dense(toVec(r, size))))
    val index = model.predict(parsedData.map(_.features))
    val predictionAndLabel = index.zip(parsedData.map(_.label))
    val result = sqlContext.createDataFrame(predictionAndLabel).toDF("pre", "real")
    //    val schemaString = "label"
    //    val schema = StructType(schemaString.split(" ").map(a => StructField(a, IntegerType, true)))
    //    val rowRDD = index.map(Row(_))
    //    val predict = sqlContext.createDataFrame(rowRDD, schema)
    result
  }

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
    //val training = fileDataFrame.rdd.map(r => LabeledPoint(r.getAs[Double]("label"), Vectors.dense(r.get(0).asInstanceOf[Double], r.get(1).asInstanceOf[Double])))

    val fileData1 = sc.textFile(path1)
    val rddRow1 = fileData1.map(_.split(",")).map(f => Row(f(0).toDouble, f(1).toDouble, f(2).toDouble))
    val schemaString1 = "c1 c2 label"
    val schema1 = StructType(schemaString1.split(" ").map(a => StructField(a, DoubleType, true)))
    val fileDataFrame1 = sqlContext.createDataFrame(rddRow1, schema1)
    //val testing = fileDataFrame1.rdd.map(r => LabeledPoint(r.getAs[Double]("label"), Vectors.dense(r.get(0).asInstanceOf[Double], r.get(1).asInstanceOf[Double])))
    //val testing = training
    val numIterations = 1000
    val modelPath = "E://svmMdl//"
    val svmIns = new SVM(sc, numIterations)
    val model = svmIns.svmTrain(fileDataFrame, modelPath)
    val result = svmIns.svmPredict(model, fileDataFrame1)
    val model1 = svmIns.svmLoad(modelPath)
    val result1 = svmIns.svmPredict(model1, fileDataFrame1)
    result.show()
    result1.show()
  }
}