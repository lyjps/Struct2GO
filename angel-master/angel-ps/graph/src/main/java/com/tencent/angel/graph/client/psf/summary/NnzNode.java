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
package com.tencent.angel.graph.client.psf.summary;

import com.tencent.angel.ml.matrix.psf.aggr.enhance.UnaryAggrFunc;
import com.tencent.angel.ps.storage.vector.ServerRow;

public class NnzNode extends UnaryAggrFunc {

    public NnzNode(int matrixId, int rowId) {
        super(matrixId, rowId);
    }

    public NnzNode() {
        super(-1, -1);
    }

    @Override
    public double mergeInit() {
        return 0;
    }

    @Override
    public double mergeOp(double a, double b) {
        return a + b;
    }

    @Override
    public double processRow(ServerRow row) {
        return row.size();
    }
}