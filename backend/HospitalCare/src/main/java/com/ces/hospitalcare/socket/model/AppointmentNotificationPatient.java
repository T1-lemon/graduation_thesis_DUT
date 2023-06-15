package com.ces.hospitalcare.socket.model;
import java.util.Date;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;

@NoArgsConstructor
@AllArgsConstructor
@Getter
@Setter
@ToString
public class AppointmentNotificationPatient {
  private Long patientId;

  private String doctorName;

  private String avatarDoctor;

  private Date startTime;

  private Integer duration;

  private Integer status;

  private Long appointmentId;

  private boolean isRead;

  private Date modifiedDate;
}
