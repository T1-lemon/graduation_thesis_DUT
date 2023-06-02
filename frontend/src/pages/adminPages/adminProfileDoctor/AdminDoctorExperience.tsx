import React, { useState } from 'react'

import ButtonCustomize from '../../../components/ButtonCustomize'
import { ProgressListener } from '../../../components/Progress'
import { AdminMedicalExaminationInterface } from '../../../interface/AdminInformationInterface'
import { useAppDispatch } from '../../../redux/hooks'
import { editMedicalExamination } from '../../../redux/thunk/adminThunk/adminDoctorThunk'
import AdminDoctorOverviewTextArea from './AdminDoctorOverviewTextArea'

type Props = {
  medicalExamination?: AdminMedicalExaminationInterface
}

function AdminDoctorExperience(props: Props) {
  const { medicalExamination } = props
  const dispatch = useAppDispatch()

  const [description, setDescription] = useState(
    medicalExamination?.description
  )
  const handleChangeTextArea = (
    event: React.ChangeEvent<HTMLTextAreaElement>
  ) => {
    setDescription(event.target.value)
  }

  const handleBtnSubmit = () => {
    if (medicalExamination) {
      const medicalExaminationData = {
        ...medicalExamination,
        description: description ? description : medicalExamination.description
      }
      const fetchApiEditMedicalExamination = async () => {
        ProgressListener.emit('start')
        await dispatch(editMedicalExamination(medicalExaminationData))
        ProgressListener.emit('stop')
      }

      fetchApiEditMedicalExamination()
    }
  }

  return (
    <div className='admin__doctor--experience'>
      <AdminDoctorOverviewTextArea
        des={medicalExamination?.description}
        onChangeTextArea={handleChangeTextArea}
        rows={15}
        title='Description'
      />
      <div className='admim__btn--overview'>
        <ButtonCustomize
          text='Save'
          className='btn__department--add'
          onClickBtn={handleBtnSubmit}
        />
      </div>
    </div>
  )
}

export default AdminDoctorExperience
