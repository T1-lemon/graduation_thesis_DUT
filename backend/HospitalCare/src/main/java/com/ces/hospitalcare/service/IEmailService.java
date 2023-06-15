package com.ces.hospitalcare.service;
import com.ces.hospitalcare.entity.UserEntity;
import jakarta.mail.MessagingException;
import java.io.UnsupportedEncodingException;
import java.util.Date;

public interface IEmailService {
  void sendEmail(String toEmail, String subject, String body) throws MessagingException;
  String messageReminderBody(String doctorName, String patientName, Date appointmentDate);
  String messageDoctorCancelAppointmentBody(String doctorName, String patientName, Date appointmentDate);
  String messageDoctorAcceptAppointmentBody(String doctorName, String patientName, Date appointmentDate);
  String messageSubject(String doctorName, String titleSubject);
  void sendVerificationEmail(UserEntity user, String siteURL, String verifyToken)
      throws MessagingException, UnsupportedEncodingException;
}
