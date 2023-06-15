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
public class AppointmentDoctorNotification {
  private Long patientId;

  private Long doctorId;

  private String avatarPatient;

  private Date startTime;

  private Integer duration;

  private String patientName;

  private Date createdDate;

  private boolean isRead;
}
