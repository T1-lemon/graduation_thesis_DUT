package com.ces.hospitalcare.http.response;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class NotificationPatientResponse {
  private Boolean isRead;

  private AppointmentResponse inforNotification;
}
