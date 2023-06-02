package com.ces.hospitalcare.repository;
import com.ces.hospitalcare.entity.ConversationEntity;
import java.util.List;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface ConversationRepository extends JpaRepository<ConversationEntity, Long> {
  List<ConversationEntity> getAllBySenderId(Long senderId);
}
