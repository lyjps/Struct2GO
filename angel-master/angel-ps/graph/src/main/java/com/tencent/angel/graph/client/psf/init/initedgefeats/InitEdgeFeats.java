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

package com.tencent.angel.graph.client.psf.init.initedgefeats;

import com.tencent.angel.graph.client.psf.init.GeneralInitParam;
import com.tencent.angel.graph.data.GraphNode;
import com.tencent.angel.graph.data.LongEdgeFeats;
import com.tencent.angel.graph.utils.GraphMatrixUtils;
import com.tencent.angel.ml.matrix.psf.update.base.GeneralPartUpdateParam;
import com.tencent.angel.ml.matrix.psf.update.base.PartitionUpdateParam;
import com.tencent.angel.ml.matrix.psf.update.base.UpdateFunc;
import com.tencent.angel.ps.storage.vector.ServerLongAnyRow;
import com.tencent.angel.ps.storage.vector.element.IElement;
import com.tencent.angel.psagent.matrix.transport.router.operator.ILongKeyAnyValuePartOp;

/**
 * Init neighbor table for GraphNode
 */
public class InitEdgeFeats extends UpdateFunc {

  /**
   * Create a new UpdateParam
   */
  public InitEdgeFeats(GeneralInitParam param) {
    super(param);
  }

  public InitEdgeFeats() {
    this(null);
  }

  @Override
  public void partitionUpdate(PartitionUpdateParam partParam) {
    GeneralPartUpdateParam initParam = (GeneralPartUpdateParam) partParam;
    ServerLongAnyRow row = GraphMatrixUtils.getPSLongKeyRow(psContext, initParam);

    // Get nodes and features
    ILongKeyAnyValuePartOp split = (ILongKeyAnyValuePartOp) initParam.getKeyValuePart();
    long[] nodeIds = split.getKeys();
    IElement[] edgeFeatures = split.getValues();

    row.startWrite();
    try {
      for(int i = 0; i < nodeIds.length; i++) {
        GraphNode graphNode = (GraphNode) row.get(nodeIds[i]);
        if (graphNode == null) {
          graphNode = new GraphNode();
          row.set(nodeIds[i], graphNode);
        }
        graphNode.setNeighbors(((LongEdgeFeats) edgeFeatures[i]).getNodeIds());
        graphNode.setEdgeFeatures(((LongEdgeFeats) edgeFeatures[i]).getFeats());
      }
    } finally {
      row.endWrite();
    }
  }
}