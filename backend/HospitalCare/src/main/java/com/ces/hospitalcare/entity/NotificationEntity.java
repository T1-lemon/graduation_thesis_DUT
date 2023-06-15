package com.ces.hospitalcare.entity;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
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
@Table(name = "Notifications")
public class NotificationEntity extends BaseEntity {
  @Column
  @Builder.Default
  private Boolean isRead = false;

  @ManyToOne
  @JoinColumn(name = "userId")
  private UserEntity user;

  @ManyToOne
  @JoinColumn(name = "appointmentId")
  private AppointmentEntity appointment;
}
