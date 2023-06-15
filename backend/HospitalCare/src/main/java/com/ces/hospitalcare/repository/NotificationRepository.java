package com.ces.hospitalcare.repository;
import com.ces.hospitalcare.entity.NotificationEntity;
import java.util.List;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

public interface NotificationRepository extends JpaRepository<NotificationEntity, Long> {
  @Query(value = "SELECT notifications.* FROM notifications\n"
      + "inner join appointments on notifications.appointment_id = appointments.id\n"
      + "WHERE notifications.user_id = :userId and (notifications.created_date between DATE_SUB(NOW(), INTERVAL 3 MONTH)  AND NOW()) \n"
      + "order by created_date desc;", nativeQuery = true)
  List<NotificationEntity> getNotificationsUserInTwoMonth(
      @Param("userId") Long userId);

  NotificationEntity findByAppointmentIdAndUserId(Long appointmentId, Long userId);
}
