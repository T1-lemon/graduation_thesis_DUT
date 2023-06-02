package com.ces.hospitalcare.repository;
import com.ces.hospitalcare.entity.TimeSlotEntity;
import java.util.List;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

@Repository
public interface TimeSlotRepository extends JpaRepository<TimeSlotEntity, Long> {
  List<TimeSlotEntity> getAllByMedicalExaminationId(Long examinationId);

  @Query(value = "SELECT * FROM time_slots WHERE doctor_id = :doctorId and YEARWEEK(start_time) = YEARWEEK(NOW());", nativeQuery = true)
  List<TimeSlotEntity> getAllTimeSlotsOfCurrentWeek(@Param("doctorId") long doctorId);

  TimeSlotEntity getByIdAndDoctorId(Long id, Long doctorId);
}
