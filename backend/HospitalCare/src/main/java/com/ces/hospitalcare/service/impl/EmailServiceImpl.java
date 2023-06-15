package com.ces.hospitalcare.service.impl;
import com.ces.hospitalcare.entity.UserEntity;
import com.ces.hospitalcare.service.IEmailService;
import jakarta.mail.MessagingException;
import jakarta.mail.internet.MimeMessage;
import java.io.UnsupportedEncodingException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.mail.javamail.MimeMessageHelper;
import org.springframework.stereotype.Service;

@Service
public class EmailServiceImpl implements IEmailService {
  @Autowired
  private JavaMailSender mailSender;
  @Override
  public void sendEmail(String toEmail, String subject, String body) throws MessagingException {
    MimeMessage message = mailSender.createMimeMessage();
    MimeMessageHelper helper = new MimeMessageHelper(message);

    helper.setSubject(subject);
    helper.setFrom("hospitalcaretech@gmail.com");
    helper.setTo(toEmail);
    helper.setText(body, true);

    mailSender.send(message);
    System.out.println("Mail Sent successfully!");
  }

  @Override
  public String messageReminderBody(String doctorName, String patientName, Date appointmentDate) {
    List<String> stringsFormatDateAndHour = dateAndHourFormat(appointmentDate);
    String formattedDate = stringsFormatDateAndHour.get(0);
    String formattedHour = stringsFormatDateAndHour.get(1);

    String message = "<b>Dear " + patientName + "</b><br></br>"
        + "This is a reminder that you have an appointment scheduled for tomorrow <b>" + formattedDate + "</b> at <b>"
        + formattedHour+"h </b>"
        + " with <b>Dr. " + doctorName +"."+"</b><br></br><br></br>Please arrive at least 10 minutes early to complete any necessary paperwork."
        + "<br></br>If you need to cancel or reschedule your appointment, please contact us as soon as possible."
        + "<br></br>Thank you for choosing our clinic for your healthcare needs."
        + "<br></br><br></br>Best regards,"
        + "<br></br>HospitalCare";

    return message;
  }

  @Override
  public String messageDoctorCancelAppointmentBody(String doctorName, String patientName, Date appointmentDate) {
    List<String> stringsFormatDateAndHour = dateAndHourFormat(appointmentDate);
    String formattedDate = stringsFormatDateAndHour.get(0);
    String formattedHour = stringsFormatDateAndHour.get(1);


    String message = "Dear<b> " + patientName + "</b><br></br><br></br>"
        + "We regret to inform you that your scheduled appointment with <b>Dr. "+ doctorName +"."
        +"</b> on <b>"+ formattedDate +"</b> at <b>"+ formattedHour +"h</b> has been <b>canceled. </b>"
        + "We apologize for any inconvenience caused.<br></br>"
        + "The cancellation may have been necessary due to unforeseen circumstances or an emergency situation that requires the doctor's immediate attention. "
        + "We understand that this may disrupt your plans, and we sincerely apologize for any inconvenience caused.<br></br><br></br>"
        + "To ensure that you receive the necessary healthcare services, we kindly request you to contact our clinic as soon as possible to reschedule your appointment. "
        + "Our staff will assist you in finding a suitable alternative date and time that fits your schedule.<br></br>"
        + "Once again, we apologize for any inconvenience caused by this cancellation. "
        + "We value your trust in our clinic and are committed to providing you with the best possible care.<br></br><br></br>"
        + "If you have any questions or need further assistance, please do not hesitate to reach out to our clinic's receptionist at the provided contact information below.<br></br>"
        + "Thank you for your understanding.<br></br><br></br>"
        + "Best regards, <br></br>"
        + "HospitalCare";
    return message;
  }

  @Override
  public String messageDoctorAcceptAppointmentBody(String doctorName, String patientName, Date appointmentDate) {
    List<String> stringsFormatDateAndHour = dateAndHourFormat(appointmentDate);
    String formattedDate = stringsFormatDateAndHour.get(0);
    String formattedHour = stringsFormatDateAndHour.get(1);


    String message = "Dear<b> " + patientName + "</b><br></br>"
    + "We are pleased to inform you that your appointment with <b>Dr. "+ doctorName +"</b> has been <b>accepted</b> and scheduled as requested. We look forward to providing you with the healthcare services you require.<br></br><br></br>"
    + "Appointment details:<br></br>"
    + "Doctor: <b>Dr. "+ doctorName +"</b>.<br></br>"
    + "Date: <b>"+ formattedDate +"</b>.<br></br>"
    + "Time: <b>"+ formattedHour +"h</b>.<br></br><br></br>"
    + "Please arrive at least 10 minutes early to complete any necessary paperwork. If there are any specific instructions or preparations you need to be aware of before your appointment, our clinic will reach out to you separately.<br></br>"
    + "If you have any questions or need to make any changes to your appointment, please feel free to contact us at the provided contact information below. We will be happy to assist you.<br></br><br></br>"
    + "Thank you for choosing our clinic for your healthcare needs. We appreciate the opportunity to serve you and look forward to seeing you on the scheduled date.<br></br><br></br>"
    + "Best regards,<br></br>"
    + "HospitalCare";
    return message;
  }

  @Override
  public String messageSubject(String doctorName, String titleSubject) {
    String message = ""+ titleSubject + doctorName;
    return message;
  }

  @Override
  public void sendVerificationEmail(UserEntity user, String siteURL, String verifyToken)
      throws MessagingException, UnsupportedEncodingException {
    String toAddress = user.getEmail();
    System.out.println(toAddress);
    String fromAddress = "hospitalcaretech@gmail.com";
    String subject = "Please verify your registration";
    String content = "Dear [[name]],<br>"
        + "Please click the link below to verify your registration:<br>"
        + "<h3><a href=\"[[URL]]\" target=\"_self\">VERIFY</a></h3>"
        + "Thank you,<br>"
        + "Your company name.";
    MimeMessage message = mailSender.createMimeMessage();
    MimeMessageHelper helper = new MimeMessageHelper(message);

    content = content.replace("[[name]]", user.getFirstName()) + " " + user.getLastName();
    String verifyURL = siteURL + "/verify/" + verifyToken;

    content = content.replace("[[URL]]", verifyURL);

    helper.setText(content, true);

    helper.setFrom(fromAddress);
    helper.setTo(toAddress);
    helper.setSubject(subject);

    mailSender.send(message);
  }

  private List<String> dateAndHourFormat(Date appointmentDate) {
    SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd");
    SimpleDateFormat hourFormat = new SimpleDateFormat("HH:mm");
    String formattedDate = dateFormat.format(appointmentDate);
    String formattedHour = hourFormat.format(appointmentDate);

    List<String> stringsFormatDateAndHour = new ArrayList<>();
    stringsFormatDateAndHour.add(formattedDate);
    stringsFormatDateAndHour.add(formattedHour);

    return stringsFormatDateAndHour;
  }
}
