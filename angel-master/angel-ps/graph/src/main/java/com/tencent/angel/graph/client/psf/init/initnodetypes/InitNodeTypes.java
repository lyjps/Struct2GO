package com.tencent.angel.graph.client.psf.init.initnodetypes;

import com.tencent.angel.graph.client.psf.init.GeneralInitParam;
import com.tencent.angel.graph.data.GraphNode;
import com.tencent.angel.graph.data.NodeType;
import com.tencent.angel.graph.utils.GraphMatrixUtils;
import com.tencent.angel.ml.matrix.psf.update.base.GeneralPartUpdateParam;
import com.tencent.angel.ml.matrix.psf.update.base.PartitionUpdateParam;
import com.tencent.angel.ml.matrix.psf.update.base.UpdateFunc;
import com.tencent.angel.ps.storage.vector.ServerLongAnyRow;
import com.tencent.angel.ps.storage.vector.element.IElement;
import com.tencent.angel.psagent.matrix.transport.router.operator.ILongKeyAnyValuePartOp;

public class InitNodeTypes extends UpdateFunc {

  /**
   * Create a new UpdateParam
   */
  public InitNodeTypes(GeneralInitParam param) {
    super(param);
  }

  public InitNodeTypes() {
    this(null);
  }

  @Override
  public void partitionUpdate(PartitionUpdateParam partParam) {
    GeneralPartUpdateParam param = (GeneralPartUpdateParam) partParam;
    ServerLongAnyRow row = GraphMatrixUtils.getPSLongKeyRow(psContext, param);
    ILongKeyAnyValuePartOp keyValuePart = (ILongKeyAnyValuePartOp) param.getKeyValuePart();

    long[] nodeIds = keyValuePart.getKeys();
    IElement[] neighbors = keyValuePart.getValues();

    row.startWrite();
    try {
      for(int i = 0; i < nodeIds.length; i++) {
        GraphNode graphNode = (GraphNode) row.get(nodeIds[i]);
        if (graphNode == null) {
          graphNode = new GraphNode();
          row.set(nodeIds[i], graphNode);
        }
        graphNode.setTypes(((NodeType) neighbors[i]).getTypes());
      }
    } finally {
      row.endWrite();
    }
  }
}
