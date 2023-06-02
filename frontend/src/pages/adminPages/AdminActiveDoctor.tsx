import React from 'react'

import AdminDoctorPage from './AdminDoctorPage'

type Props = {}

const AdminActiveDoctor = (props: Props) => {
  return (
    <>
      <AdminDoctorPage isArchive={0} />
    </>
  )
}

export default AdminActiveDoctor
