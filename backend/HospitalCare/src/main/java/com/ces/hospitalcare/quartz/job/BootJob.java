package com.ces.hospitalcare.quartz.job;
import com.ces.hospitalcare.http.response.EmailByDateResponse;
import com.ces.hospitalcare.repository.AppointmentRepository;
import com.ces.hospitalcare.service.IEmailService;
import jakarta.mail.MessagingException;
import java.time.LocalDate;
import java.util.List;
import org.quartz.Job;
import org.quartz.JobExecutionContext;
import org.quartz.JobExecutionException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

@Component
public class BootJob implements Job {
  @Autowired
  private IEmailService emailService;

  @Autowired
  private AppointmentRepository appointmentRepository;

  @Override
  public void execute(JobExecutionContext jobExecutionContext) throws JobExecutionException {
    System.out.println("Boot Job");
    LocalDate tomorrow = LocalDate.now().plusDays(1);
//    List<EmailByDateResponse> listEmailByAppointmentDate = appointmentRepository.getListEmailByAppointmentDate(
//        "2023-04-24");
    List<EmailByDateResponse> listEmailByAppointmentDate = appointmentRepository.getListEmailByAppointmentDate(
        tomorrow.toString());
    for (EmailByDateResponse emailByAppointmentDate : listEmailByAppointmentDate) {
      String doctorName = emailByAppointmentDate.getFirstNameDoctor() + " "
          + emailByAppointmentDate.getLastNameDoctor();
      String patientName = emailByAppointmentDate.getFirstNamePatient() + " "
          + emailByAppointmentDate.getLastNamePatient();
      String messageSubject = emailService.messageSubject(doctorName, "Reminder: Appointment Tomorrow with doctor ");
      String messageBody = emailService.messageReminderBody(doctorName, patientName, emailByAppointmentDate.getStartTime());
      System.out.println(patientName);
      try {
//        emailService.sendEmail("buianhtuan2111@gmail.com", messageSubject, messageBody);
        emailService.sendEmail(emailByAppointmentDate.getEmail(), messageSubject, messageBody);
      } catch (MessagingException e) {
        throw new RuntimeException(e);
      }
//      emailService.sendEmail(emailByAppointmentDate.getEmail(), messageSubject, messageBody);
    }

  }
}
