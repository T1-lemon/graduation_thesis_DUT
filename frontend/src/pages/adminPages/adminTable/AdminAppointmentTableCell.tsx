import { TableRow } from '@mui/material'

import {
  AdminTableColumn,
  TableAppointment
} from '../../../interface/AdminTableInterface'
import { TableCellProfile } from '../../../themes/profileStyle'
import { RenderStatus } from '../../patientPages/dashboardPage/TableBookedAppointment'
import AdminGroupBtnAction from './AdminGroupBtnAction'
import AdminTableApptDate from './AdminTableApptDate'
import AdminTableGroupName from './AdminTableGroupName'

interface Props {
  columns: AdminTableColumn[]
  row: TableAppointment
}

const AdminAppointmentTableCell = (props: Props) => {
  const { columns, row } = props
  return (
    <TableRow hover role='checkbox' tabIndex={-1}>
      {columns.map((column: AdminTableColumn) => {
        if (column.id === 'actions') {
          return (
            <TableCellProfile key={column.id}>
              <AdminGroupBtnAction apptId={row.id} />
            </TableCellProfile>
          )
        }
        const columnId = column.id as keyof typeof row
        const value = row[columnId]
        if (value === row.doctorName || value === row.patientName) {
          return (
            <TableCellProfile key={column.id} align={column.align}>
              <AdminTableGroupName
                profilePicture={
                  value === row.doctorName
                    ? row.profilePictureDoctor
                    : row.profilePicturePatient
                }
                name={
                  value === row.doctorName ? row.doctorName : row.patientName
                }
              />
            </TableCellProfile>
          )
        }

        if (value === row.apptDate) {
          return (
            <TableCellProfile key={column.id} align={column.align}>
              <AdminTableApptDate
                apptDate={row.apptDate}
                time={`${row.startTime} - ${row.endTime}`}
              />
            </TableCellProfile>
          )
        }

        if (value === row.status) {
          return (
            <TableCellProfile key={column.id} align={column.align}>
              <RenderStatus status={row.status} />
            </TableCellProfile>
          )
        }

        return (
          <TableCellProfile key={column.id} align={column.align}>
            {value}
          </TableCellProfile>
        )
      })}
    </TableRow>
  )
}

export default AdminAppointmentTableCell
