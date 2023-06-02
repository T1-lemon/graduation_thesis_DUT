package com.ces.hospitalcare.entity;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.OneToOne;
import jakarta.persistence.Table;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@Builder
@AllArgsConstructor
@NoArgsConstructor
@Entity
@Table(name = "Appointments")
public class AppointmentEntity extends BaseEntity {
  @Column
  @Builder.Default
  private Integer status = 0;

  @ManyToOne
  @JoinColumn(name = "patientId")
  private UserEntity patient;

  @ManyToOne
  @JoinColumn(name = "doctorId")
  private UserEntity doctor;

  @OneToOne
  @JoinColumn(name = "timeSlotId")
  private TimeSlotEntity timeSlot;
}
