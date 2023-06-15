package com.ces.hospitalcare.service;
import com.ces.hospitalcare.dto.NotificationDTO;
import com.ces.hospitalcare.http.request.NotificationRequest;
import com.ces.hospitalcare.http.response.NotificationPatientResponse;
import java.util.List;

public interface INotificationService {
  List<NotificationDTO> getNotificationsDoctor(Long doctorId);

  List<NotificationPatientResponse> getNotificationsPatient(Long patientId);

  NotificationDTO addNotification(NotificationRequest notificationRequest);

  NotificationDTO changeReadNotification(NotificationRequest notificationRequest);
}
