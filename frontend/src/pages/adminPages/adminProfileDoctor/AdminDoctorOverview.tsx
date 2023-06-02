import React, { useState } from 'react'

import ButtonCustomize from '../../../components/ButtonCustomize'
import { ProgressListener } from '../../../components/Progress'
import { AdminMedicalExaminationInterface } from '../../../interface/AdminInformationInterface'
import { useAppDispatch } from '../../../redux/hooks'
import { editMedicalExamination } from '../../../redux/thunk/adminThunk/adminDoctorThunk'
import AdminDoctorOverviewInput from './AdminDoctorOverviewInput'
import AdminDoctorOverviewTextArea from './AdminDoctorOverviewTextArea'

type Props = {
  medicalExamination?: AdminMedicalExaminationInterface
}

const AdminDoctorOverview = (props: Props) => {
  const { medicalExamination } = props
  const dispatch = useAppDispatch()

  const [title, setTitle] = useState(medicalExamination?.title)
  const [shortDescription, setSortDescription] = useState(
    medicalExamination?.shortDescription
  )

  const handleChangeTextArea = (
    event: React.ChangeEvent<HTMLTextAreaElement>
  ) => {
    setSortDescription(event.target.value)
  }

  const handleChangeTitle = (event: React.ChangeEvent<HTMLInputElement>) => {
    setTitle(event.target.value)
  }

  const handleBtnSubmit = () => {
    if (medicalExamination) {
      const medicalExaminationData = {
        ...medicalExamination,
        title: title ? title : medicalExamination.title,
        shortDescription: shortDescription
          ? shortDescription
          : medicalExamination.shortDescription
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
    <div className='admin__doctor--overview'>
      <AdminDoctorOverviewInput
        title={medicalExamination?.title}
        onChangeTitle={handleChangeTitle}
      />
      <AdminDoctorOverviewTextArea
        des={medicalExamination?.shortDescription}
        onChangeTextArea={handleChangeTextArea}
        rows={4}
        title='Short Description'
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

export default AdminDoctorOverview
