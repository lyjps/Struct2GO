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


package com.tencent.angel.master.ps;

import com.tencent.angel.ps.ParameterServerId;
import org.apache.hadoop.yarn.event.AbstractEvent;

/**
 * PS manager event.
 */
public class ParameterServerManagerEvent extends AbstractEvent<ParameterServerManagerEventType> {
  /**
   * ps id
   */
  private final ParameterServerId psId;

  /**
   * Create a ParameterServerManagerEvent
   *
   * @param type event type
   */
  public ParameterServerManagerEvent(ParameterServerManagerEventType type) {
    this(type, null);
  }

  /**
   * Create a ParameterServerManagerEvent
   *
   * @param type event type
   * @param psId ps id
   */
  public ParameterServerManagerEvent(ParameterServerManagerEventType type, ParameterServerId psId) {
    super(type);
    this.psId = psId;
  }

  /**
   * get ps id
   *
   * @return ParameterServerId ps id
   */
  public ParameterServerId getPsId() {
    return psId;
  }
}
