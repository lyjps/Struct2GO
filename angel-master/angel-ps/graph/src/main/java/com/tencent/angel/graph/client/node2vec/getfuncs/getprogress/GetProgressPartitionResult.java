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
package com.tencent.angel.graph.client.node2vec.getfuncs.getprogress;

import com.tencent.angel.common.ByteBufSerdeUtils;
import com.tencent.angel.ml.matrix.psf.get.base.PartitionGetResult;
import io.netty.buffer.ByteBuf;

public class GetProgressPartitionResult extends PartitionGetResult {

  private boolean finished;
  private double percent;

  public GetProgressPartitionResult(boolean finished, double precent) {
    this.finished = finished;
    this.percent = precent;
  }

  public GetProgressPartitionResult() {
    super();
  }

  public boolean isFinished() {
    return finished;
  }

  public void setFinished(boolean finished) {
    this.finished = finished;
  }

  public double getPrecent() {
    return percent;
  }

  public void setPrecent(double precent) {
    this.percent = precent;
  }

  @Override
  public void serialize(ByteBuf output) {
    output.writeBoolean(finished);
    output.writeDouble(percent);
  }

  @Override
  public void deserialize(ByteBuf input) {
    finished = input.readBoolean();
    percent = input.readDouble();
  }

  @Override
  public int bufferLen() {
    return ByteBufSerdeUtils.serializedBooleanLen(finished) + 8;
  }
}
