import { Avatar, TableRow } from '@mui/material'
import { useNavigate } from 'react-router-dom'

import { ProgressListener } from '../../../components/Progress'
import {
  AdminTableColumn,
  TableDepartment
} from '../../../interface/AdminTableInterface'
import { useAppDispatch } from '../../../redux/hooks'
import { deleteDepartmentByAdmin } from '../../../redux/thunk/adminThunk/adminDepartmentThunk'
import { archiveDoctorByAdminService } from '../../../services/adminServices/adminDoctorService'
import { TableCellProfile } from '../../../themes/profileStyle'
import AdminGroupBtnAction from './AdminGroupBtnAction'

interface Props {
  columns: AdminTableColumn[]
  row: TableDepartment
}

const AdminDepartmentTableCell = (props: Props) => {
  const { columns, row } = props

  return (
    <TableRow hover role='checkbox' tabIndex={-1}>
      {columns.map((column: AdminTableColumn) => {
        if (column.id === 'actions') {
          return (
            <TableCellProfile key={column.id}>
              <AdminGroupBtnAction
                departmentName={row.departmentName}
                file={row.departmentPicture}
                departmentId={row.id}
              />
            </TableCellProfile>
          )
        }
        const columnId = column.id as keyof typeof row
        const value = row[columnId]
        return (
          <TableCellProfile key={column.id} align={column.align}>
            {value === row.departmentName ? (
              <>
                <div className='department__group--name'>
                  <div className='department__picture'>
                    <Avatar src={row.departmentPicture} />
                  </div>
                  <div className='department__name'>{row.departmentName}</div>
                </div>
              </>
            ) : (
              value
            )}
          </TableCellProfile>
        )
      })}
    </TableRow>
  )
}

interface FooterModalDeleteInterface {
  departmentId?: number
  handleClose: () => void
  doctorId?: number
}
export const FooterModalDelete = (props: FooterModalDeleteInterface) => {
  const { departmentId, handleClose, doctorId } = props

  const dipatch = useAppDispatch()
  const navigate = useNavigate()

  const handleDeleteDepartment = async () => {
    if (departmentId) {
      ProgressListener.emit('start')
      await dipatch(deleteDepartmentByAdmin(departmentId as number))
      ProgressListener.emit('stop')
    }

    if (doctorId) {
      ProgressListener.emit('start')
      await archiveDoctorByAdminService(doctorId as number)
      ProgressListener.emit('stop')
      navigate('/admin/doctors/archive')
    }
  }

  return (
    <div className='footer__group--delete'>
      <button className='admin__btn admin__btn--cancle' onClick={handleClose}>
        Cancle
      </button>
      <button
        className='admin__btn admin__btn-delete'
        onClick={handleDeleteDepartment}
      >
        Delete
      </button>
    </div>
  )
}

export default AdminDepartmentTableCell
