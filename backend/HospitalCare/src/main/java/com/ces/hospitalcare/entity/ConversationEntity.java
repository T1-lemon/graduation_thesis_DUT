package com.ces.hospitalcare.entity;
import jakarta.persistence.Entity;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.OneToMany;
import jakarta.persistence.Table;
import java.util.ArrayList;
import java.util.List;
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
@Table(name = "Conversations")
public class ConversationEntity extends BaseEntity {
  @OneToMany(mappedBy = "conversation")
  private List<MessageEntity> messageList = new ArrayList<>();

  @ManyToOne
  @JoinColumn(name = "senderId")
  private UserEntity sender;

  @ManyToOne
  @JoinColumn(name = "receiverId")
  private UserEntity receiver;
}
