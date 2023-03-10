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

package com.tencent.angel.graph.client.psf.sample.samplenodefeats;

import com.tencent.angel.ml.math2.vector.IntFloatVector;
import com.tencent.angel.ml.matrix.psf.get.base.GetResult;
import java.util.Collections;
import java.util.List;

public class SampleNodeFeatResult extends GetResult {

  /**
   * randomly sampled node features array
   */
  public List<IntFloatVector> feats;

  SampleNodeFeatResult(List<IntFloatVector> feats) {
    this.feats = feats;
  }

  SampleNodeFeatResult() {
    this(null);
  }

  public IntFloatVector[] getNodeFeats() {
    Collections.shuffle(feats);
    IntFloatVector[] res = new IntFloatVector[feats.size()];
    return feats.toArray(res);
  }
}