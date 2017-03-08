package rhh

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD

import java.util.Arrays

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.ml.feature.{MinHashLSH,Tokenizer,HashingTF,Word2Vec}
import org.apache.spark.sql.functions.{concat,lit}
import org.apache.spark.sql.{SparkSession,Row}

object Similarity{

val spark = SparkSession.builder.appName("Word2Vec example").getOrCreate()

def main(args:Array[String]) = {
  //doWordCount()
  doMinhashComparison()
}

def doMinhashComparison() = {
  val datasrc = "/home/rholder/quora/src/main/resources/JEOPARDY_QUESTIONS1.json"
  val jeopardy = spark.read.json(datasrc)
  jeopardy.show()

  import spark.sqlContext.implicits._
  val jeopardyText = jeopardy.select(concat($"question",lit(" "), $"category").alias("text"))
  jeopardyText.show()

  val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("tokens")
  val vectorizer = new HashingTF().setInputCol("tokens").setOutputCol("mh_vector").setNumFeatures(1e5.toInt)
  val mh = new MinHashLSH().setNumHashTables(3).setInputCol("mh_vector").setOutputCol("values")

  val vectors = vectorizer.transform(tokenizer.transform(jeopardyText))
  vectors.show()

  val model = mh.fit(vectors)

  val dataA = vectors.sample(false,0.8)
  val dataB = vectors.sample(false,0.2)	
  val maxDf = 0.6
  val transformedA = model.transform(dataA).cache()
  val transformedB = model.transform(dataB).cache()
  val mhResults = model.approxSimilarityJoin(transformedA, transformedB, maxDf).cache()
  val mhResultsNz = mhResults.filter("distCol > 0")
  mhResultsNz.show(100,80)
}

//adapted from https://github.com/apache/spark/blob/master/examples/src/main/scala/org/apache/spark/examples/ml/Word2VecExample.scala
def doWord2VecComparison() = {

  // Input data: Each row is a bag of words from a sentence or document.
  val documentDF = spark.createDataFrame(Seq(
  "Hi I heard about Spark".split(" "),
  "I wish Java could use case classes".split(" "),
  "Logistic regression models are neat".split(" ")
  ).map(Tuple1.apply)).toDF("text")

  // Learn a mapping from words to Vectors.
  val word2Vec = new Word2Vec()
                 .setInputCol("text")
		 .setOutputCol("result")
		 .setVectorSize(3)
		 .setMinCount(0)
  val model = word2Vec.fit(documentDF)
  val result = model.transform(documentDF)
  result.collect().foreach { case Row(text: Seq[_], features: Vector) =>
  println(s"Text: [${text.mkString(", ")}] => \nVector: $features\n") }

  spark.stop()
}

def doWordCount() = {
  val sc = new SparkContext(new SparkConf().setAppName("similarity"))
  val textfile:RDD[String] = sc.textFile("file:///home/rholder/quora/src/main/resources/quora_duplicate_questions.tsv");
  val counts = textfile.flatMap(line => line.split("[\\t\\s]"))
                       .map(word => (word,1))
		       .reduceByKey(_+_)
		       .sortBy(p => p._2, false)  //tuples are 1-based indexes
		       .take(10)

  println("*************************")
  println("***** RESULTS ***********")
  println("*************************")
  counts.foreach(println)
}  

}