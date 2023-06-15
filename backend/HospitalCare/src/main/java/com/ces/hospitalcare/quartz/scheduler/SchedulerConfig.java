package com.ces.hospitalcare.quartz.scheduler;
import org.quartz.JobDetail;
import org.quartz.Scheduler;
import org.quartz.SchedulerException;
import org.quartz.SchedulerFactory;
import org.quartz.Trigger;
import org.quartz.impl.StdSchedulerFactory;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.context.annotation.Bean;

public class SchedulerConfig {
  private final JobDetail jobDetail;
  private final Trigger trigger;
  public SchedulerConfig(@Qualifier("jobDetail") JobDetail jobDetail,@Qualifier("jobTriggerBoot") Trigger trigger) {
    this.jobDetail = jobDetail;
    this.trigger = trigger;
  }
  @Bean
  public void initialJob() {
    SchedulerFactory schedulerFactory = new StdSchedulerFactory();
    try {
      Scheduler scheduler = schedulerFactory.getScheduler();
      scheduler.scheduleJob(jobDetail, trigger);
      scheduler.start();
    } catch (SchedulerException e) {
      e.printStackTrace();
    }
  }
}

