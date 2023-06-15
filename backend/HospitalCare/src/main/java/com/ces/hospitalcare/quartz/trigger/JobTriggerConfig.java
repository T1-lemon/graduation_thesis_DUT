package com.ces.hospitalcare.quartz.trigger;
import org.quartz.CronScheduleBuilder;
import org.quartz.JobDetail;
import org.quartz.Trigger;
import org.quartz.TriggerBuilder;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class JobTriggerConfig {
  @Bean(name="jobTriggerBoot")
  public Trigger jobTriggerBoot(@Qualifier("jobDetail") JobDetail jobDetail) {
    try {
      String time = "0/5 * * * * ?"; //5s
      return TriggerBuilder.newTrigger().forJob(jobDetail).withIdentity("RUN_QUARTZ", "JOB_GROUP")
          .startNow()
//          .withSchedule(CronScheduleBuilder.cronSchedule(time))
          .withSchedule(CronScheduleBuilder.dailyAtHourAndMinute(8,0))
          .build();
    } catch (Exception e) {
      System.out.println("Error trigger" + e);
      return null;
    }
  }
}