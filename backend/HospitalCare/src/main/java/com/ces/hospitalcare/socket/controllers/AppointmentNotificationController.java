package com.ces.hospitalcare.socket.controllers;
import com.ces.hospitalcare.socket.model.AppointmentDoctorNotification;
import com.ces.hospitalcare.socket.model.AppointmentNotificationPatient;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.messaging.handler.annotation.MessageMapping;
import org.springframework.messaging.handler.annotation.Payload;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.stereotype.Controller;

@Controller
public class AppointmentNotificationController {
  @Autowired
  private SimpMessagingTemplate simpMessagingTemplate;

  //  messagemapping là đường link để client send
  @MessageMapping("/private-appointment")
  public AppointmentDoctorNotification receiveNotificationAppointmentPending(
      @Payload AppointmentDoctorNotification appointmentDoctorNotification) {

    Long doctorId = appointmentDoctorNotification.getDoctorId();
    simpMessagingTemplate.convertAndSendToUser(Long.toString(doctorId),
        "/appointment",
        appointmentDoctorNotification);
    System.out.println(appointmentDoctorNotification.toString());
    return appointmentDoctorNotification;
  }

  @MessageMapping("/change-statusAppointment")
  public AppointmentNotificationPatient notificationChangeStatusAppointment(
      @Payload AppointmentNotificationPatient appointmentNotificationPatient) {

    Long patientId = appointmentNotificationPatient.getPatientId();
    simpMessagingTemplate.convertAndSendToUser(Long.toString(patientId),
        "/status-appointment",
        appointmentNotificationPatient);
    System.out.println(appointmentNotificationPatient.toString());
    return appointmentNotificationPatient;
  }
}
