package com.ces.hospitalcare.quartz.detail;
import com.ces.hospitalcare.quartz.job.BootJob;
import org.quartz.JobBuilder;
import org.quartz.JobDetail;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class JobDetailConfig {
  @Bean(name = "jobDetail")
  public JobDetail jobDetail() {
    return JobBuilder.newJob().ofType(BootJob.class)
        .withIdentity("RUN_QUARTZ", "JOB_GROUP")
        .withDescription("description job ...")
        .storeDurably(true)
        .build();
  }
}
