import DeleteIcon from '@mui/icons-material/Delete'
import EditIcon from '@mui/icons-material/Edit'
import RemoveRedEyeIcon from '@mui/icons-material/RemoveRedEye'
import { Button } from '@mui/material'
import React, { useState } from 'react'

import { useAppSelector } from '../../../redux/hooks'
import AdminModelCustomize from '../../../utils/models/AdminModelCustomize'
import ModalAddDepartmentBody from '../../../utils/models/ModalAddDepartmentBody'
import AdminAppointmentDetail from '../adminAppointment/AdminAppointmentDetail'
import AdminPatientDetail from '../adminPatient/AdminPatientDetail'
import { FooterModalDelete } from './AdminDepartmentTableCell'

type Props = {
  departmentName?: string
  apptId?: number
  patientId?: number
  file?: File | string
  departmentId?: number
}

interface openModelDepartment {
  openEdit: boolean
  openDelete: boolean
  openView: boolean
}

const AdminGroupBtnAction = (props: Props) => {
  const { departmentName, apptId, patientId, file, departmentId } = props
  const { messageDeleteDepartment } = useAppSelector((state) => state.admin)

  const [open, setOpen] = useState<openModelDepartment>({
    openEdit: false,
    openDelete: false,
    openView: false
  })

  const handleOpen = (status?: string) => {
    status === 'edit'
      ? setOpen({
          ...open,
          openEdit: true
        })
      : status === 'delete'
      ? setOpen({
          ...open,
          openDelete: true
        })
      : setOpen({
          ...open,
          openView: true
        })
  }

  const handleClose = (status?: string) => {
    status === 'edit'
      ? setOpen({
          ...open,
          openEdit: false
        })
      : status === 'delete'
      ? setOpen({
          ...open,
          openDelete: false
        })
      : setOpen({
          ...open,
          openView: false
        })
  }
  return (
    <>
      <div className='div__group--action'>
        {departmentName ? (
          <Button
            onClick={() => handleOpen('edit')}
            variant='contained'
            className='btn__edit btn__group--action'
            startIcon={<EditIcon />}
          />
        ) : (
          <></>
        )}

        {departmentName ? (
          <Button
            onClick={() => handleOpen('delete')}
            variant='contained'
            className='btn__delete btn__group--action'
            startIcon={<DeleteIcon />}
          />
        ) : (
          <></>
        )}

        {apptId || patientId ? (
          <Button
            onClick={() => handleOpen('view')}
            variant='contained'
            className='btn__view btn__group--action'
            startIcon={<RemoveRedEyeIcon />}
          />
        ) : (
          <></>
        )}
      </div>

      <div>
        {departmentName && file ? (
          <>
            <AdminModelCustomize
              classNameHeader='success'
              title='Edit Department'
              open={open.openEdit}
              handleClose={() => handleClose('edit')}
              bodyModal={
                <ModalAddDepartmentBody
                  inputEdit={departmentName}
                  file={file}
                  id={departmentId}
                  handleClose={() => handleClose('edit')}
                />
              }
            />
            <AdminModelCustomize
              classNameHeader='danger'
              title='Delete Department'
              open={open.openDelete}
              handleClose={() => handleClose('delete')}
              titleDelete='You are about to delete this department'
              desDelete={
                messageDeleteDepartment
                  ? messageDeleteDepartment
                  : 'This will delete this department. Are you sure?'
              }
              footerModal={
                <FooterModalDelete
                  departmentId={departmentId}
                  handleClose={() => handleClose('delete')}
                />
              }
            />
          </>
        ) : (
          <></>
        )}

        {apptId ? (
          <AdminModelCustomize
            classNameHeader='success'
            classNameContainer='admin__modal--detailAppt'
            title='Detail Appointment'
            open={open.openView}
            handleClose={() => handleClose('view')}
            bodyModal={<AdminAppointmentDetail apptId={apptId} />}
          />
        ) : (
          <></>
        )}

        {patientId ? (
          <AdminModelCustomize
            classNameHeader='success'
            classNameContainer='admin__modal--detailPatient'
            title='Detail Information Of Patient'
            open={open.openView}
            handleClose={() => handleClose('view')}
            bodyModal={<AdminPatientDetail patientId={patientId} />}
          />
        ) : (
          <></>
        )}
      </div>
    </>
  )
}

export default AdminGroupBtnAction
