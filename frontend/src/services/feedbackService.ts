import requestApi from '../utils/requestApi/requestApi';
import requestAuthApi from '../utils/requestApi/requestAuthApi';


export const addFeedbackService = async (
  examinationId: number,
  commentText: string
) => {
  try {
    const response = await requestAuthApi({
      url: `/feedbacks`,
      method: 'post',
      data: {
        examinationId,
        commentText
      }
    })
    return response.data
  } catch (error) {
    return error
  }
}

export const getAllFeedbackMedicalService = async (examinationId: number) => {
  try {
    const response = await requestApi({
      url: `/feedbacks?examinationId=${examinationId}`,
      method: 'get'
    })
    return response.data
  } catch (error) {
    return error
  }
}