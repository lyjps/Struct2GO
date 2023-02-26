package com.tencent.angel.spark.examples.local
import java.io.File

  /**
    * 20170309
    * 目录操作
    */

  object dir {

    def main(args: Array[String]) {
      println("hahahaha")
      val path = new File("/Users/lyjps/Desktop/科研/人类蛋白质数据集/proteins_edges")
      for(name<- getFile(path)){
        println(name)
      }
//      val path: File = new File("/Users/lyjps/Desktop/科研/人类蛋白质数据集/proteins_edges")
//      for (d <- subdirs(path)) {
//        println(d)
//        println("haha")
//      }
//
//      println(path)
    }

    //遍历目录

    def getFile(file:File): Array[File] ={
      val files = file.listFiles().filter(! _.isDirectory)
        .filter(t => t.toString.endsWith(".txt") || t.toString.endsWith(".md"))  //此处读取.txt and .md文件
      files ++ file.listFiles().filter(_.isDirectory).flatMap(getFile)
    }

  }

