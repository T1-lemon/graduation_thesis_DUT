package com.ces.hospitalcare.repository;
import com.ces.hospitalcare.entity.MessageEntity;
import java.util.List;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

@Repository
public interface MessageRepository extends JpaRepository<MessageEntity, Long> {
  @Query(
      value =
          "select messages.* from messages\n"
              + "inner join conversations on conversations.id = messages.conversation_id\n"
              + "where (sender_id = :senderId and receiver_id =:receiverId) or (sender_id = :receiverId and receiver_id = :senderId)\n"
              + "order by messages.created_date;",
      nativeQuery = true)
  List<MessageEntity> getAllMessageTwoUser(
      @Param("senderId") Long senderId, @Param("receiverId") Long receiverId);
}


