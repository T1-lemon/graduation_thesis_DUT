import React, { useEffect, useState } from 'react'
import Moment from 'react-moment'
import { toast } from 'react-toastify'

import ButtonCustomize from '../../../components/ButtonCustomize'
import {
  ICheckConditionFeedback,
  IFeedback
} from '../../../interface/FeedbackInterface'
import { IMedicalExamination } from '../../../interface/MedicalExaminationInterfaces'
import { useAppSelector } from '../../../redux/hooks'
import { checkAppointmentPatientAndExaminationService } from '../../../services/appointmentService'
import {
  addFeedbackService,
  getAllFeedbackMedicalService
} from '../../../services/feedbackService'

type Props = {
  medical?: IMedicalExamination
}
const logo = require('../../../assets/img/vietnam_flag.png')

export default function ReviewExamination(props: Props) {
  const { medical } = props
  const { currentUser, isAuth } = useAppSelector((state) => state.auths)
  const [isBookMedical, setIsBookMedical] = useState<boolean>()
  const [feedbacks, setFeedbacks] = useState<IFeedback[]>([])
  const [inputFeedback, setInputFeedback] = useState<string>('')

  useEffect(() => {
    getAllFeedbacks()
    checkBookExaminate()
  }, [medical])

  const getAllFeedbacks = async () => {
    if (medical?.id) {
      const data = await getAllFeedbackMedicalService(medical?.id as number)
      setFeedbacks(data)
    }
  }

  const renderFeedback = () => {
    if (feedbacks.length > 0) {
      return feedbacks.map((feedback) => {
        const { patient, commentText, createdDate } = feedback
        return (
          <div className='review_item'>
            <div className='avatar_patient'>
              <img src={patient.profilePicture} alt='' />
            </div>
            <div className='review_infor-container'>
              <strong>{patient.firstName + ' ' + patient.lastName}</strong>
              <div className='date_review'>
                <Moment fromNow>{createdDate}</Moment>
              </div>
              <p className='comment_text'>{commentText}</p>
            </div>
          </div>
        )
      })
    }
  }

  const checkBookExaminate = async () => {
    if (isAuth) {
      const data: ICheckConditionFeedback =
        await checkAppointmentPatientAndExaminationService(
          medical?.id as number
        )
      // console.log('check book', data)
      setIsBookMedical(data.booked)
    }
  }

  const onChangeTextarea = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputFeedback(event.target.value)
  }

  const addFeedback = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!isBookMedical) {
      toast.warn('You have not booked an appointment for this doctor yet')

      return
    }

    // console.log('add feedback')
    if (medical) {
      await addFeedbackService(medical.id, inputFeedback)
      getAllFeedbacks()
      setInputFeedback('')
    }
  }

  return (
    <>
      <div className='reviews_container'>{renderFeedback()}</div>
      {isAuth && isBookMedical ? (
        <div className='input_review'>
          <img src={currentUser.profilePicture} alt='' />
          <form action='' onSubmit={addFeedback}>
            <textarea
              name=''
              id='textarea_review'
              placeholder='Your review'
              onChange={onChangeTextarea}
              value={inputFeedback}
            ></textarea>
            <ButtonCustomize
              text='Add comment'
              className='btn__comment'
              type='submit'
            />
          </form>
        </div>
      ) : (
        ''
      )}
    </>
  )
}
