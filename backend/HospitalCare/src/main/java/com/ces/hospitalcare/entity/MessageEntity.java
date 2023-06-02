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
@Table(name = "Messages")
public class MessageEntity extends BaseEntity {
  @Column(columnDefinition = "TEXT")
  private String text;

  @ManyToOne
  @JoinColumn(name = "conversationId")
  private ConversationEntity conversation;
}
