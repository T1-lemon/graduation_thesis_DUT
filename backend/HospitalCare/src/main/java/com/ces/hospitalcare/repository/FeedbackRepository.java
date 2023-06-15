package com.ces.hospitalcare.repository;
import com.ces.hospitalcare.entity.FeedbackEntity;
import java.util.List;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface FeedbackRepository extends JpaRepository<FeedbackEntity, Long> {
  List<FeedbackEntity> getAllByMedicalExaminationIdOrderByCreatedDateDesc(Long examinationId);
}
