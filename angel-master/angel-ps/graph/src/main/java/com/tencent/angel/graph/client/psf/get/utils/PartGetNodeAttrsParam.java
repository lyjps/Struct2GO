/*
 * Tencent is pleased to support the open source community by making Angel available.
 *
 * Copyright (C) 2017-2018 THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/Apache-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 */
package com.tencent.angel.graph.client.psf.get.utils;

import com.tencent.angel.PartitionKey;
import com.tencent.angel.common.ByteBufSerdeUtils;
import com.tencent.angel.ml.matrix.psf.get.base.PartitionGetParam;
import io.netty.buffer.ByteBuf;

public class PartGetNodeAttrsParam extends PartitionGetParam {

  /**
   * Node ids
   */
  private long[] nodeIds;

  private int startIndex;
  private int endIndex;


  public PartGetNodeAttrsParam(int matrixId, PartitionKey part, long[] nodeIds
      , int startIndex, int endIndex) {
    super(matrixId, part);
    this.nodeIds = nodeIds;
    this.startIndex = startIndex;
    this.endIndex = endIndex;
  }

  public PartGetNodeAttrsParam() {
    this(-1, null, null, -1, -1);
  }

  public long[] getNodeIds() {
    return nodeIds;
  }

  public void setNodeIds(long[] nodeIds) {
    this.nodeIds = nodeIds;
  }

  public int getStartIndex() {
    return startIndex;
  }

  public int getEndIndex() {
    return endIndex;
  }

  @Override
  public void serialize(ByteBuf buf) {
    super.serialize(buf);
    ByteBufSerdeUtils.serializeInt(buf, endIndex - startIndex);
    for (int i = startIndex; i < endIndex; i++) {
      ByteBufSerdeUtils.serializeLong(buf, nodeIds[i]);
    }
  }

  @Override
  public void deserialize(ByteBuf buf) {
    super.deserialize(buf);
    nodeIds = new long[ByteBufSerdeUtils.deserializeInt(buf)];
    for (int i = 0; i < nodeIds.length; i++) {
      nodeIds[i] = ByteBufSerdeUtils.deserializeLong(buf);
    }
  }

  @Override
  public int bufferLen() {
    return super.bufferLen() + ByteBufSerdeUtils.INT_LENGTH +
        ByteBufSerdeUtils.LONG_LENGTH * (endIndex - startIndex);
  }
}