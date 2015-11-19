/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.horn.trainer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hama.Constants;
import org.apache.hama.HamaCluster;
import org.apache.hama.HamaConfiguration;
import org.apache.hama.bsp.BSPJob;
import org.apache.hama.bsp.HashPartitioner;
import org.apache.hama.bsp.NullOutputFormat;
import org.apache.hama.bsp.TextInputFormat;

public class TestTrainer extends HamaCluster {
  protected HamaConfiguration configuration;

  // these variables are preventing from rebooting the whole stuff again since
  // setup and teardown are called per method.

  public TestTrainer() {
    configuration = new HamaConfiguration();
    configuration.set("bsp.master.address", "localhost");
    configuration.set("hama.child.redirect.log.console", "true");
    assertEquals("Make sure master addr is set to localhost:", "localhost",
        configuration.get("bsp.master.address"));
    configuration.set("bsp.local.dir", "/tmp/hama-test");
    configuration.set(Constants.ZOOKEEPER_QUORUM, "localhost");
    configuration.setInt(Constants.ZOOKEEPER_CLIENT_PORT, 21810);
    configuration.set("hama.sync.client.class",
        org.apache.hama.bsp.sync.ZooKeeperSyncClientImpl.class
            .getCanonicalName());
  }

  @Override
  public void setUp() throws Exception {
    super.setUp();
  }

  @Override
  public void tearDown() throws Exception {
    super.tearDown();
  }

  public void testOutputJob() throws Exception {
    String strTrainingDataPath = "src/test/resources/neuralnets_classification_training.txt";

    Configuration conf = new Configuration();
    conf.set("bsp.local.dir", "/tmp/hama-test");
    conf.setInt("horn.max.iteration", 100);
    conf.setInt("horn.minibatch.size", 10);
    conf.setBoolean("bsp.input.runtime.partitioning", true);

    BSPJob bsp = new BSPJob(new HamaConfiguration(conf));
    bsp.setJobName("Test Replica Trainer");

    bsp.setPartitioner(HashPartitioner.class);

    bsp.setBspClass(Trainer.class);
    bsp.setOutputFormat(NullOutputFormat.class);

    bsp.setNumBspTask(2);
    bsp.setInputFormat(TextInputFormat.class);
    bsp.setInputPath(new Path(strTrainingDataPath));

    bsp.waitForCompletion(true);
  }

}
